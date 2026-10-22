"""Kawkab AI Cloud Server — FastAPI application."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from kawkab.api.api_v1 import router as api_v1_router
from kawkab.cloud.auth import (
    create_access_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from kawkab.cloud.database import get_cloud_db
from kawkab.cloud.middleware import RateLimitMiddleware
from kawkab.cloud.models import (
    ConflictRecord,
    OAuthAuthorizeResponse,
    OAuthCallbackRequest,
    PasswordChange,
    SharedProject,
    SyncOperation,
    SyncPayload,
    SyncResponse,
    TeamCreate,
    TeamInvite,
    TokenResponse,
    UserLogin,
    UserOut,
    UserRegister,
)
from kawkab.services.storage_errors import StorageError

app = FastAPI(title="Kawkab AI Cloud", version="0.1.0")


@app.exception_handler(StorageError)
async def storage_error_handler(request: Request, exc: StorageError) -> JSONResponse:
    """Honest storage failures map to 503 (backend unavailable), not 500.

    StorageNotInitializedError means the backend never came up; other
    StorageErrors are per-operation DB failures. Both are retryable
    server-side conditions, so 503 with the typed operation name is the
    honest API shape.
    """
    status = 503
    return JSONResponse(
        status_code=status,
        content={
            "error": "storage_unavailable",
            # Only the operation-carrying StorageError subclasses expose
            # .operation; the bare base does not — "unknown" is the honest
            # fallback for a typed failure without an operation name.
            "operation": getattr(exc, "operation", "unknown"),
            "detail": str(exc.__cause__ or exc),
        },
    )


app.include_router(api_v1_router)

_cors_origins = os.environ.get("KAWKAB_CORS_ORIGINS", "https://app.kawkab.ai").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)  # type: ignore[arg-type]


# ── Health ──

import platform
import time

_start_time = time.time()


@app.get("/health")
def health():
    db_ok = False
    try:
        from kawkab.cloud.database import get_cloud_db

        db = get_cloud_db()
        db.execute("SELECT 1")
        db_ok = True
    except Exception:
        pass
    return {
        "status": "ok" if db_ok else "degraded",
        "version": "0.1.0",
        "uptime_s": round(time.time() - _start_time, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "database": "connected" if db_ok else "unreachable",
    }


@app.get("/health/ready")
def health_ready():
    try:
        from kawkab.cloud.database import get_cloud_db

        db = get_cloud_db()
        db.execute("SELECT 1")
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "error": str(e)}


@app.get("/health/live")
def health_live():
    return {"alive": True}


@app.get("/metrics")
def metrics():
    import gc

    return {
        "uptime_s": round(time.time() - _start_time, 1),
        "python_version": platform.python_version(),
        "gc_count": gc.get_count(),
        "gc_threshold": list(gc.get_threshold()),
    }


# ── Auth ──


@app.post("/auth/register", response_model=TokenResponse)
def register(body: UserRegister):
    db = get_cloud_db()
    existing = db.execute(
        "SELECT id FROM users WHERE email = ? OR username = ?", (body.email, body.username)
    ).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Email or username already taken")
    pwd_hash = hash_password(body.password)
    role = getattr(body, "role", "analyst")
    if role not in ("admin", "coach", "analyst", "scout", "viewer"):
        role = "analyst"
    cur = db.execute(
        "INSERT INTO users (username, email, password_hash, display_name, role) VALUES (?, ?, ?, ?, ?)",
        (body.username, body.email, pwd_hash, body.display_name or body.username, role),
    )
    db.commit()
    user_id = cur.lastrowid
    user_row = db.execute(
        "SELECT id, username, email, display_name, role, is_active, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    user = cast(dict[str, Any], dict(user_row or {}))
    assert user_id is not None
    token = create_access_token(int(user_id), role=role)
    return TokenResponse(access_token=token, user=UserOut(**user))


@app.post("/auth/login", response_model=TokenResponse)
def login(body: UserLogin):
    db = get_cloud_db()
    row = db.execute("SELECT * FROM users WHERE email = ?", (body.email,)).fetchone()
    if not row or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    # row.get(...) used to be called directly on `row` here. On the SQLite
    # backend `row` is a sqlite3.Row, which supports row["role"] but has no
    # .get() method at all -- so every login on the default (non-Postgres)
    # backend raised AttributeError -> HTTP 500, with no test covering
    # /auth/login anywhere to catch it. Converting to a plain dict first
    # works identically on both backends (_ResultRow on Postgres already
    # is a dict subclass).
    user = dict(row)
    token = create_access_token(
        user["id"], role=user.get("role", "analyst"), token_version=user.get("token_version", 0)
    )
    del user["password_hash"]
    return TokenResponse(access_token=token, user=UserOut(**user))


@app.get("/auth/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)):
    return UserOut(**user)


@app.post("/auth/change-password")
def change_password(body: PasswordChange, user: dict = Depends(get_current_user)):
    db = get_cloud_db()
    row = db.execute(
        "SELECT password_hash, token_version FROM users WHERE id = ?", (user["id"],)
    ).fetchone()
    assert row is not None  # user id comes from the verified access token
    if not verify_password(body.old_password, row["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    # Bumping token_version invalidates every token issued before this
    # change -- including the one used to make this very request -- which
    # is the whole point: a stolen token shouldn't survive its victim
    # changing their password. A fresh token is returned below so the
    # caller isn't forced to immediately log back in.
    new_token_version = row["token_version"] + 1
    db.execute(
        "UPDATE users SET password_hash = ?, token_version = ?, updated_at = datetime('now') WHERE id = ?",
        (hash_password(body.new_password), new_token_version, user["id"]),
    )
    db.commit()
    new_token = create_access_token(
        user["id"], role=user.get("role", "analyst"), token_version=new_token_version
    )
    return {"ok": True, "access_token": new_token}


# ── OAuth ──

import contextlib
import secrets

from kawkab.cloud.oauth import get_configured_providers, get_oauth_provider

_oauth_states: dict[str, str] = {}  # state -> provider


@app.get("/auth/oauth/providers")
def oauth_providers():
    return {"providers": get_configured_providers()}


@app.get("/auth/oauth/{provider}/authorize", response_model=OAuthAuthorizeResponse)
def oauth_authorize(provider: str, redirect_uri: str = ""):
    prov = get_oauth_provider(provider)
    if not prov:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    state = secrets.token_hex(16)
    _oauth_states[state] = provider
    url = prov.get_authorize_url(redirect_uri or f"/auth/oauth/{provider}/callback", state)
    return OAuthAuthorizeResponse(authorize_url=url, state=state, provider=provider)


@app.post("/auth/oauth/{provider}/callback", response_model=TokenResponse)
def oauth_callback(provider: str, body: OAuthCallbackRequest):
    prov = get_oauth_provider(provider)
    if not prov:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    expected_provider = _oauth_states.pop(body.state, None)
    if expected_provider != provider:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    tokens = prov.exchange_code(body.code, f"/auth/oauth/{provider}/callback")
    if tokens is None:
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code")
    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="No access token in response")
    userinfo = prov.get_userinfo(access_token)
    if userinfo is None:
        raise HTTPException(status_code=400, detail="Failed to fetch user info")
    provider_user_id = str(userinfo.get("id") or userinfo.get("sub"))
    email = userinfo.get("email", "")
    name = userinfo.get("name") or userinfo.get("login") or email.split("@")[0]
    db = get_cloud_db()
    row = db.execute(
        "SELECT user_id FROM oauth_accounts WHERE provider = ? AND provider_user_id = ?",
        (provider, provider_user_id),
    ).fetchone()
    if row:
        db.execute(
            "UPDATE oauth_accounts SET access_token = ?, refresh_token = ? WHERE id = ?",
            (access_token, tokens.get("refresh_token"), row["user_id"]),
        )
        db.commit()
        _rows: Any = db.execute(
            "SELECT id, username, email, display_name, is_active, token_version, created_at FROM users WHERE id = ?",
            (row["user_id"],),
        ).fetchone()
        user = cast(dict[str, Any], dict(_rows))
        # token_version must come from the DB, not default to 0: a
        # returning OAuth user may have had it bumped since (e.g. a
        # password change on a linked local-auth account), and minting a
        # token with the wrong version would lock them out immediately.
        jwt_token = create_access_token(user["id"], token_version=user.get("token_version", 0))
        return TokenResponse(access_token=jwt_token, user=UserOut(**user))
    existing_user = None
    if email:
        existing_user = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if existing_user:
        user_id = existing_user["id"]
    else:
        username = name.lower().replace(" ", "_")
        base = username
        counter = 1
        while db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone():
            username = f"{base}_{counter}"
            counter += 1
        cur = db.execute(
            "INSERT INTO users (username, email, password_hash, display_name) VALUES (?, ?, ?, ?)",
            (username, email or f"{provider}_{provider_user_id}@oauth.local", "oauth", name),
        )
        db.commit()
        user_id = cur.lastrowid
    db.execute(
        "INSERT INTO oauth_accounts (user_id, provider, provider_user_id, access_token, refresh_token) VALUES (?, ?, ?, ?, ?)",
        (user_id, provider, provider_user_id, access_token, tokens.get("refresh_token")),
    )
    db.commit()
    _rows = db.execute(
        "SELECT id, username, email, display_name, is_active, token_version, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    user = cast(dict[str, Any], dict(_rows))
    jwt_token = create_access_token(user_id, token_version=user.get("token_version", 0))
    return TokenResponse(access_token=jwt_token, user=UserOut(**user))


@app.get("/auth/oauth/accounts")
def list_oauth_accounts(user: dict = Depends(get_current_user)):
    db = get_cloud_db()
    rows = db.execute(
        "SELECT provider, provider_user_id, created_at FROM oauth_accounts WHERE user_id = ?",
        (user["id"],),
    ).fetchall()
    return [dict(r) for r in rows]


@app.post("/auth/oauth/unlink")
def unlink_oauth_account(provider: str, user: dict = Depends(get_current_user)):
    db = get_cloud_db()
    password_row = db.execute(
        "SELECT password_hash FROM users WHERE id = ?", (user["id"],)
    ).fetchone()
    assert password_row is not None  # same-token guarantee
    if password_row["password_hash"] == "oauth":
        count_row = db.execute(
            "SELECT COUNT(*) as c FROM oauth_accounts WHERE user_id = ?", (user["id"],)
        ).fetchone()
        assert count_row is not None  # COUNT(*) always returns a row
        pw_count = count_row["c"]
        if pw_count <= 1:
            raise HTTPException(
                status_code=400, detail="Cannot unlink last login method. Set a password first."
            )
    db.execute(
        "DELETE FROM oauth_accounts WHERE user_id = ? AND provider = ?", (user["id"], provider)
    )
    db.commit()
    return {"ok": True}


# ── SAML SSO (Phase 4: wires the previously-orphaned saml_auth_service.py) ──


class SAMLCallbackRequest(BaseModel):
    saml_response_xml: str
    idp_entity_id: str


class SAMLConfigIn(BaseModel):
    entity_id: str
    sso_url: str
    certificate: str = ""


_saml_states: dict[str, str] = {}  # relay_state -> idp_entity_id


def _get_saml_service():
    from kawkab.services.saml_auth_service import SAMLAuthService

    return SAMLAuthService()


@app.get("/auth/saml/status")
def saml_status():
    """Whether the SAML SDK is importable and which IdPs are configured.

    Registration lives in process memory (same lifetime as the OAuth
    state map) — a deployment wires its IdP once at boot via
    KAWKAB_SAML_IDP_ENTITY_ID / KAWKAB_SAML_IDP_SSO_URL / KAWKAB_SAML_IDP_CERT.
    """
    svc = _get_saml_service()
    if not svc.available and os.environ.get("KAWKAB_SAML_IDP_SSO_URL"):
        svc.register_idp(
            type(svc).SAMLIdentityProvider(
                entity_id=os.environ.get("KAWKAB_SAML_IDP_ENTITY_ID", "club-idp"),
                sso_url=os.environ["KAWKAB_SAML_IDP_SSO_URL"],
                certificate=os.environ.get("KAWKAB_SAML_IDP_CERT", ""),
            )
        )
    return {
        "sdk_available": svc.available,
        "configured_idps": [i.entity_id for i in svc.get_configured_providers()],
    }


@app.post("/auth/saml/configure")
def saml_configure(body: SAMLConfigIn, user: dict = Depends(get_current_user)):
    """Register/replace a SAML IdP (admin only — the club IT dept's step)."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    svc = _get_saml_service()
    if not svc.available:
        raise HTTPException(
            status_code=503,
            detail="SAML SDK not installed (pip install python3-saml or onelogin-graphql-sdk alternative)",
        )
    from kawkab.services.saml_auth_service import SAMLIdentityProvider

    svc.register_idp(
        SAMLIdentityProvider(
            entity_id=body.entity_id,
            sso_url=body.sso_url,
            certificate=body.certificate,
        )
    )
    return {"ok": True, "entity_id": body.entity_id}


@app.post("/auth/saml/callback", response_model=TokenResponse)
def saml_callback(body: SAMLCallbackRequest):
    """Consume the IdP's SAMLResponse POST, resolve/create the local user,
    and mint the same JWT the password/OAuth flows return."""
    svc = _get_saml_service()
    if not svc.available:
        raise HTTPException(status_code=503, detail="SAML SDK not available on this deployment")
    result = svc.handle_acs(body.saml_response_xml)
    if not result or not result.get("email"):
        raise HTTPException(status_code=400, detail="Invalid SAML response (no assertion/email)")

    email = result["email"]
    db = get_cloud_db()
    row = db.execute(
        "SELECT id, username, email, display_name, is_active, token_version, created_at FROM users WHERE email = ?",
        (email,),
    ).fetchone()
    if row:
        user = dict(row)
        if not user.get("is_active", 1):
            raise HTTPException(status_code=403, detail="Account disabled")
        jwt_token = create_access_token(user["id"], token_version=user.get("token_version", 0))
        return TokenResponse(access_token=jwt_token, user=UserOut(**user))

    # First SSO login: provision the account (analyst by default — the
    # cloud server's other provisioning paths use the same default).
    username = email.split("@")[0].lower().replace(" ", "_")
    base = username
    counter = 1
    while db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone():
        username = f"{base}_{counter}"
        counter += 1
    cur = db.execute(
        "INSERT INTO users (username, email, password_hash, display_name) VALUES (?, ?, ?, ?)",
        (username, email, "saml", result.get("name") or username),
    )
    db.commit()
    _rows: Any = db.execute(
        "SELECT id, username, email, display_name, is_active, token_version, created_at FROM users WHERE id = ?",
        (cur.lastrowid,),
    ).fetchone()
    user = cast(dict[str, Any], dict(_rows))
    jwt_token = create_access_token(user["id"], token_version=user.get("token_version", 0))
    return TokenResponse(access_token=jwt_token, user=UserOut(**user))


# ── Sync ──


@app.post("/sync/push")
def sync_push(payload: SyncPayload, user: dict = Depends(get_current_user)):
    db = get_cloud_db()
    conflicts = []
    applied = []
    for op in payload.operations:
        existing = db.execute(
            "SELECT version, data FROM projects WHERE id = ? AND owner_id = ?",
            (op.entity_id, user["id"]),
        ).fetchone()

        # `existing` above is scoped to *this user's own* rows, so it comes
        # back None both for "brand new project id" and for "a project with
        # this id exists but is owned by someone else" -- previously those
        # two cases were indistinguishable. A cross-user id collision on
        # create/update fell through to the ON CONFLICT(id) DO UPDATE below,
        # which has no owner check at all: any authenticated user submitting
        # someone else's real project id got their data/version silently
        # written over the victim's project (owner_id itself wasn't
        # touched, but the content was destroyed). Distinguish the two
        # cases explicitly before ever reaching the upsert.
        if existing is None and op.op in ("create", "update"):
            owned_by_other = db.execute(
                "SELECT owner_id FROM projects WHERE id = ? AND owner_id != ?",
                (op.entity_id, user["id"]),
            ).fetchone()
            if owned_by_other is not None:
                conflicts.append(
                    ConflictRecord(
                        entity_type=op.entity_type,
                        entity_id=op.entity_id,
                        local_version=int(op.data.get("_version", 0)),
                        remote_version=-1,
                        local_data=op.data,
                        remote_data={"error": "This id belongs to a project you do not own."},
                    )
                )
                continue

        if existing and op.op == "update":
            local_ver = int(op.data.get("_version", 0))
            if local_ver < existing["version"]:
                conflicts.append(
                    ConflictRecord(
                        entity_type=op.entity_type,
                        entity_id=op.entity_id,
                        local_version=local_ver,
                        remote_version=existing["version"],
                        local_data=op.data,
                        remote_data=json.loads(existing["data"]),
                    )
                )
                continue
        if op.op == "delete":
            db.execute(
                "DELETE FROM projects WHERE id = ? AND owner_id = ?", (op.entity_id, user["id"])
            )
            db.execute(
                "INSERT INTO sync_log (user_id, device_id, entity_type, entity_id, operation) VALUES (?,?,?,?,?)",
                (user["id"], payload.device_id, op.entity_type, op.entity_id, "delete"),
            )
            applied.append(op)
        elif op.op in ("create", "update"):
            data_json = json.dumps(op.data, ensure_ascii=False)
            new_ver = existing["version"] + 1 if existing else 1
            # The WHERE on DO UPDATE is defense-in-depth on top of the
            # explicit ownership check above: even if that check were ever
            # bypassed or buggy, SQLite itself refuses to apply the update
            # when the existing row's owner_id doesn't match, rather than
            # silently overwriting another user's row.
            db.execute(
                """INSERT INTO projects (id, name, owner_id, data, version)
                          VALUES (?,?,?,?,?)
                          ON CONFLICT(id) DO UPDATE SET data=excluded.data, version=excluded.version, updated_at=datetime('now')
                          WHERE projects.owner_id = excluded.owner_id""",
                (op.entity_id, op.data.get("name", "Untitled"), user["id"], data_json, new_ver),
            )
            db.execute(
                "INSERT INTO sync_log (user_id, device_id, entity_type, entity_id, operation) VALUES (?,?,?,?,?)",
                (user["id"], payload.device_id, op.entity_type, op.entity_id, op.op),
            )
            applied.append(op)
    db.commit()
    return SyncResponse(
        sync_token=str(datetime.now(UTC).timestamp()),
        operations=applied,
        conflicts=conflicts,
    )


@app.post("/sync/pull")
def sync_pull(payload: SyncPayload, user: dict = Depends(get_current_user)):
    db = get_cloud_db()
    rows = db.execute(
        "SELECT id, name, data, version, updated_at FROM projects WHERE owner_id = ? ORDER BY updated_at DESC LIMIT 100",
        (user["id"],),
    ).fetchall()
    ops = []
    for r in rows:
        data = json.loads(r["data"])
        data["_version"] = r["version"]
        ops.append(SyncOperation(op="update", entity_type="project", entity_id=r["id"], data=data))
    return SyncResponse(sync_token=str(len(rows)), operations=ops)


# ── Teams ──


@app.post("/teams")
def create_team(body: TeamCreate, user: dict = Depends(get_current_user)):
    db = get_cloud_db()
    cur = db.execute(
        "INSERT INTO teams (name, description, owner_id) VALUES (?, ?, ?)",
        (body.name, body.description, user["id"]),
    )
    db.execute(
        "INSERT INTO team_members (team_id, user_id, role) VALUES (?, ?, 'owner')",
        (cur.lastrowid, user["id"]),
    )
    db.commit()
    return {"ok": True, "team_id": cur.lastrowid}


@app.get("/teams")
def list_teams(user: dict = Depends(get_current_user)):
    db = get_cloud_db()
    rows = db.execute(
        """SELECT t.*, tm.role FROM teams t
                         JOIN team_members tm ON t.id = tm.team_id
                         WHERE tm.user_id = ? ORDER BY t.name""",
        (user["id"],),
    ).fetchall()
    return [dict(r) for r in rows]


@app.post("/teams/{team_id}/invite")
def invite_member(team_id: int, body: TeamInvite, user: dict = Depends(get_current_user)):
    db = get_cloud_db()
    owner = db.execute(
        "SELECT role FROM team_members WHERE team_id = ? AND user_id = ?", (team_id, user["id"])
    ).fetchone()
    if not owner or owner["role"] not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Not authorized")
    token = str(uuid.uuid4())
    db.execute(
        "INSERT INTO team_invites (team_id, email, role, token) VALUES (?, ?, ?, ?)",
        (team_id, body.email, body.role, token),
    )
    db.commit()
    return {"ok": True, "invite_token": token}


@app.post("/teams/join/{token}")
def accept_invite(token: str, user: dict = Depends(get_current_user)):
    db = get_cloud_db()
    invite = db.execute(
        "SELECT * FROM team_invites WHERE token = ? AND accepted = 0", (token,)
    ).fetchone()
    if not invite:
        raise HTTPException(status_code=404, detail="Invalid or expired invite")
    db.execute(
        "INSERT OR IGNORE INTO team_members (team_id, user_id, role) VALUES (?, ?, ?)",
        (invite["team_id"], user["id"], invite["role"]),
    )
    db.execute("UPDATE team_invites SET accepted = 1 WHERE id = ?", (invite["id"],))
    db.commit()
    return {"ok": True, "team_id": invite["team_id"]}


# ── Shared Projects ──


@app.post("/projects/share")
def share_project(body: SharedProject, user: dict = Depends(get_current_user)):
    db = get_cloud_db()
    project = db.execute("SELECT * FROM projects WHERE id = ?", (body.project_id,)).fetchone()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project["owner_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    db.execute(
        "UPDATE projects SET is_shared = 1, team_id = ? WHERE id = ?",
        (body.team_id, body.project_id),
    )
    db.commit()
    return {"ok": True}


# ── WebSocket for real-time collaboration ──

connected_clients: dict[str, list[WebSocket]] = {}


def _authorize_project_access(db, project_id: str, user_id: int) -> bool:
    """True if user_id may join a collaboration session for project_id.

    Authorized if the user owns the project outright, or the project is
    marked shared and the user belongs to the team it's shared with.
    """
    project = db.execute(
        "SELECT owner_id, team_id, is_shared FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    if project is None:
        return False
    if project["owner_id"] == user_id:
        return True
    if project["is_shared"] and project["team_id"] is not None:
        member = db.execute(
            "SELECT 1 FROM team_members WHERE team_id = ? AND user_id = ?",
            (project["team_id"], user_id),
        ).fetchone()
        return member is not None
    return False


@app.websocket("/ws/{project_id}")
async def ws_endpoint(websocket: WebSocket, project_id: str, token: str = Query(...)):
    """Real-time collaboration channel for one project.

    Previously had no authentication at all -- no token, no origin
    check, no project-membership check. Any internet client could
    connect to any project_id, read every collaborator's live edits,
    and inject arbitrary messages into their session. Browsers can't
    set a custom Authorization header on a WebSocket handshake, so the
    JWT is passed as a query parameter instead (matches how most
    WebSocket APIs handle this); everything is rejected with an explicit
    close code *before* accept() so a rejected client never enters the
    broadcast group.
    """
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in _cors_origins and "*" not in _cors_origins:
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    payload = decode_token(token)
    if payload is None:
        await websocket.close(code=1008, reason="Invalid or missing token")
        return
    user_id = int(payload["sub"])

    db = get_cloud_db()
    user_row = db.execute("SELECT is_active FROM users WHERE id = ?", (user_id,)).fetchone()
    if user_row is None or not user_row["is_active"]:
        await websocket.close(code=1008, reason="User not found or inactive")
        return

    if not _authorize_project_access(db, project_id, user_id):
        await websocket.close(code=1008, reason="Not authorized for this project")
        return

    await websocket.accept()
    if project_id not in connected_clients:
        connected_clients[project_id] = []
    connected_clients[project_id].append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Broadcast to other clients in the same project
            for client in connected_clients.get(project_id, []):
                if client != websocket:
                    with contextlib.suppress(Exception):
                        await client.send_text(data)
    except WebSocketDisconnect:
        connected_clients[project_id] = [
            c for c in connected_clients.get(project_id, []) if c != websocket
        ]


# ── Run ──


def start(host: str = "0.0.0.0", port: int = 8741):
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start()
