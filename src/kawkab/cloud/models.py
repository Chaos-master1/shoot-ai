from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

# ── Auth ──


class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str = ""


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    display_name: str
    is_active: bool
    created_at: datetime
    # Every construction site does `UserOut(**user)` from a full-row dict
    # that already has `role` (login/register/me/change-password all
    # `SELECT *`). Without this field pydantic silently dropped it from
    # every response -- an RBAC-based API returning a token but never
    # telling the client what role it authenticated as. Defaulted (not
    # required) because the two OAuth callback sites still `SELECT` an
    # explicit column list that omits `role` -- see cloud/server.py's
    # oauth_callback -- so they fall back to "analyst" rather than
    # erroring. That fallback can be wrong for a non-analyst OAuth user;
    # the real fix is adding `role` to those two SELECTs too.
    role: str = "analyst"


class PasswordChange(BaseModel):
    old_password: str
    new_password: str = Field(min_length=8)


# ── Sync ──


class SyncPayload(BaseModel):
    device_id: str
    last_sync_at: str | None = None
    operations: list[SyncOperation] = []


class SyncOperation(BaseModel):
    op: str  # "create" | "update" | "delete"
    entity_type: str  # "match" | "project" | "scout_report" | "tag"
    entity_id: str
    data: dict = {}


class SyncResponse(BaseModel):
    sync_token: str
    operations: list[SyncOperation] = []
    conflicts: list[ConflictRecord] = []


class ConflictRecord(BaseModel):
    entity_type: str
    entity_id: str
    local_version: int
    remote_version: int
    local_data: dict
    remote_data: dict


# ── Team ──


class TeamCreate(BaseModel):
    name: str
    description: str = ""


class TeamMember(BaseModel):
    user_id: int
    role: str = "member"  # "owner" | "admin" | "member" | "viewer"


class TeamInvite(BaseModel):
    email: EmailStr
    role: str = "member"


class SharedProject(BaseModel):
    project_id: str
    team_id: int
    permissions: str = "view"  # "view" | "edit" | "admin"


# ── OAuth ──


class OAuthAuthorizeResponse(BaseModel):
    authorize_url: str
    state: str
    provider: str


class OAuthCallbackRequest(BaseModel):
    code: str
    state: str
    provider: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


# ── WebSocket ──


class WSMessage(BaseModel):
    type: str  # "edit" | "cursor" | "comment" | "presence"
    project_id: str
    user_id: int
    payload: dict = {}
    timestamp: float = 0.0
