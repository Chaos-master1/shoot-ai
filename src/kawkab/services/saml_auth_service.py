from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SAMLServiceProvider:
    entity_id: str
    acs_url: str
    certificate: str = ""
    issuer: str = ""


@dataclass
class SAMLIdentityProvider:
    entity_id: str
    sso_url: str
    certificate: str = ""
    name_id_format: str = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"


class SAMLAuthService:
    def __init__(self) -> None:
        self._available = False
        self._onelogin = None
        self._idps: dict[str, SAMLIdentityProvider] = {}
        self._sps: dict[str, SAMLServiceProvider] = {}
        self._try_load()

    def _try_load(self) -> None:
        try:
            from onelogin.saml2.auth import OneLogin_Saml2_Auth

            self._onelogin = OneLogin_Saml2_Auth
            self._available = True
            logger.info("OneLogin SAML SDK loaded")
        except Exception as exc:
            logger.info(f"OneLogin SAML SDK not available: {exc}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def initiate_sso(self, sp_entity_id: str, idp_entity_id: str, relay_state: str = "") -> str:
        if not self._available:
            logger.warning("SAML not available")
            return ""
        sp = self._sps.get(sp_entity_id)
        idp = self._idps.get(idp_entity_id)
        if not sp or not idp:
            logger.warning(f"Unknown SP ({sp_entity_id}) or IdP ({idp_entity_id})")
            return ""
        try:
            return self._build_authn_request(sp, idp, relay_state)
        except Exception as exc:
            logger.warning(f"Failed to build SAML request: {exc}")
            return ""

    def handle_acs(self, response_xml: str) -> dict[str, Any]:
        if not self._available or not response_xml:
            logger.warning("SAML not available or empty response")
            return {}
        try:
            return self._parse_saml_response(response_xml)
        except Exception as exc:
            logger.warning(f"Failed to parse SAML response: {exc}")
            return {}

    def get_configured_providers(self) -> list[SAMLIdentityProvider]:
        return list(self._idps.values())

    def register_idp(self, config: SAMLIdentityProvider) -> None:
        self._idps[config.entity_id] = config
        logger.info(f"Registered SAML IdP: {config.entity_id}")

    def register_sp(self, config: SAMLServiceProvider) -> None:
        self._sps[config.entity_id] = config
        logger.info(f"Registered SAML SP: {config.entity_id}")

    def _build_authn_request(
        self, sp: SAMLServiceProvider, idp: SAMLIdentityProvider, relay_state: str
    ) -> str:
        _ = sp.acs_url.rstrip("/")
        saml_request = (
            f'<samlp:AuthnRequest xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"'
            f' xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"'
            f' ID="_{sp.entity_id}" Version="2.0"'
            f' IssueInstant="{self._now_iso()}"'
            f' Destination="{idp.sso_url}"'
            f' AssertionConsumerServiceURL="{sp.acs_url}"'
            f' ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">'
            f"<saml:Issuer>{sp.entity_id}</saml:Issuer>"
            f'<samlp:NameIDPolicy Format="{idp.name_id_format}" AllowCreate="true"/>'
            f"</samlp:AuthnRequest>"
        )
        import base64
        import urllib.parse
        import zlib

        deflated = zlib.compress(saml_request.encode("utf-8"))
        encoded = base64.b64encode(deflated).decode("utf-8")
        url = f"{idp.sso_url}?SAMLRequest={urllib.parse.quote(encoded)}"
        if relay_state:
            url += f"&RelayState={urllib.parse.quote(relay_state)}"
        return url

    def _parse_saml_response(self, response_xml: str) -> dict[str, Any]:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response_xml)
        ns = {
            "samlp": "urn:oasis:names:tc:SAML:2.0:protocol",
            "saml": "urn:oasis:names:tc:SAML:2.0:assertion",
        }
        assertion = root.find(".//saml:Assertion", ns)
        if assertion is None:
            return {}
        attr_stmt = assertion.find(".//saml:AttributeStatement", ns)
        attributes: dict[str, str] = {}
        if attr_stmt is not None:
            for attr in attr_stmt.findall("saml:Attribute", ns):
                name = attr.get("Name", "")
                vals = [v.text or "" for v in attr.findall("saml:AttributeValue", ns)]
                if vals:
                    attributes[name] = vals[0]
        subject = assertion.find(".//saml:Subject/saml:NameID", ns)
        email = subject.text if subject is not None else ""
        return {
            "email": email,
            "name": attributes.get("displayName", attributes.get("cn", email)),
            "attributes": attributes,
        }

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime

        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
