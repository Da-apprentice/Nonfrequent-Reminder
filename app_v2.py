"""
Insurance Renewal Guardian — Version 2
Renewal-focused: AI analysis, smart escalating reminders, payment confirmation.
"""

import json
import os
import re
import smtplib
import sqlite3
import traceback
import uuid
import xml.sax.saxutils
from datetime import date, datetime, time, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any
from urllib.parse import quote

import google.generativeai as genai
import pdfplumber
import streamlit as st
from dotenv import dotenv_values

# Secrets file — NOT named .env (avoids editor caching and auto-loading by other tools)
ENV_FILE_NAME = "guardian_secrets.env"
APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ENV_FILE_NAME

CONFIG_KEYS = (
    "GEMINI_MODEL",
    "GEMINI_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_WHATSAPP_FROM",
    "TWILIO_SMS_FROM",
    "TWILIO_VOICE_FROM",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USER",
    "SMTP_PASSWORD",
    "SMTP_FROM",
    "SMTP_SSL",
)

ENV_PLACEHOLDERS = {
    "[authtoken]",
    "your_auth_token",
    "your_real_auth_token_here",
    "your_real_32_character_token_here",
    "your_account_sid",
    "your_twilio_auth_token",
    "your_twilio_account_sid",
    "acxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
}


def clean_env_value(value: str) -> str:
    """Strip whitespace and surrounding quotes from an env value."""
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "'\"":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def is_valid_env_value(value: str) -> bool:
    """True when a value is present and not a documentation placeholder."""
    if not value:
        return False
    return value.strip().lower() not in ENV_PLACEHOLDERS


def read_env_file_text(path: Path) -> tuple[str, str]:
    """Read .env bytes from disk, handling Notepad UTF-16 saves on Windows."""
    if not path.exists():
        return "", "missing"
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le"), "UTF-16 LE"
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be"), "UTF-16 BE"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "UTF-8"
    if len(raw) >= 2 and raw[1:2] == b"\x00":
        return raw.decode("utf-16-le"), "UTF-16 LE (no BOM)"
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8 (fallback)"


def parse_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE pairs directly from disk (fresh read every call)."""
    text, _encoding = read_env_file_text(path)
    if not text:
        return {}
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("export "):
            stripped = stripped[7:].strip()
        if "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        key = key.strip()
        if not key:
            continue
        parsed[key] = clean_env_value(raw_value)
    return parsed


def parse_env_file_with_dotenv(path: Path) -> dict[str, str]:
    """Merge our disk parser with python-dotenv for edge cases; disk parser wins."""
    if not path.exists():
        return {}
    merged = dict(dotenv_values(path))
    merged.update(parse_env_file(path))
    return merged


def get_streamlit_secret(key: str) -> str:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            value = clean_env_value(str(st.secrets[key]))
            if value:
                return value
    except Exception:
        pass
    return ""


def get_env_file_stats() -> dict[str, Any]:
    """Safe metadata about the .env file on disk (no secret values)."""
    if not ENV_PATH.exists():
        return {
            "exists": False,
            "path": str(ENV_PATH),
            "size_bytes": 0,
            "modified": "",
            "encoding": "",
        }
    stat = ENV_PATH.stat()
    _text, encoding = read_env_file_text(ENV_PATH)
    return {
        "exists": True,
        "path": str(ENV_PATH),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "encoding": encoding,
    }


def get_config_source(key: str, file_values: dict[str, str], resolved: str) -> str:
    if clean_env_value(file_values.get(key, "") or ""):
        return "file"
    if get_streamlit_secret(key):
        return "secrets"
    if not ENV_PATH.exists() and clean_env_value(os.getenv(key, "") or ""):
        return "os.environ"
    if resolved:
        return "default"
    return "missing"


def get_stale_env_note(key: str, file_values: dict[str, str]) -> str | None:
    """Detect when a cached OS env var differs from what is on disk."""
    file_val = clean_env_value(file_values.get(key, "") or "")
    os_val = clean_env_value(os.getenv(key, "") or "")
    if not os_val:
        return None
    if file_val and file_val != os_val:
        return (
            f"`{key}`: cached OS env ({len(os_val)} chars) ignored — "
            f"using .env file ({len(file_val)} chars)"
        )
    if ENV_PATH.exists() and not file_val and os_val:
        return (
            f"`{key}`: OS env has {len(os_val)} chars but .env parse found nothing — "
            "try Save As UTF-8 in Notepad"
        )
    return None


def load_config() -> dict[str, str]:
    """Load config from guardian_secrets.env on disk. Ignores stale os.environ and .env."""
    file_values = parse_env_file_with_dotenv(ENV_PATH) if ENV_PATH.exists() else {}
    # Drop cached OS copies so nothing else reads stale Twilio/Gemini values
    if file_values:
        for key in CONFIG_KEYS:
            os.environ.pop(key, None)

    def pick(key: str, default: str = "") -> str:
        file_val = clean_env_value(file_values.get(key, "") or "")
        if file_val:
            return file_val
        secret = get_streamlit_secret(key)
        if secret:
            return secret
        if not ENV_PATH.exists():
            return clean_env_value(os.getenv(key, default))
        return clean_env_value(default)

    return {
        "GEMINI_MODEL": pick("GEMINI_MODEL", "gemini-flash-latest"),
        "GEMINI_API_KEY": pick("GEMINI_API_KEY"),
        "TWILIO_ACCOUNT_SID": pick("TWILIO_ACCOUNT_SID"),
        "TWILIO_AUTH_TOKEN": pick("TWILIO_AUTH_TOKEN"),
        "TWILIO_WHATSAPP_FROM": pick("TWILIO_WHATSAPP_FROM"),
        "TWILIO_SMS_FROM": pick("TWILIO_SMS_FROM"),
        "TWILIO_VOICE_FROM": pick("TWILIO_VOICE_FROM"),
        "SMTP_HOST": pick("SMTP_HOST"),
        "SMTP_PORT": pick("SMTP_PORT", "465"),
        "SMTP_USER": pick("SMTP_USER"),
        "SMTP_PASSWORD": pick("SMTP_PASSWORD"),
        "SMTP_FROM": pick("SMTP_FROM"),
        "SMTP_SSL": pick("SMTP_SSL", "true"),
    }


def get_manual_twilio_token() -> str:
    """Token typed in the UI for this session (overrides .env placeholder)."""
    for key in ("manual_twilio_auth_token", "manual_twilio_auth_token_input"):
        value = st.session_state.get(key, "")
        if isinstance(value, str) and value.strip():
            cleaned = clean_env_value(value.strip())
            if cleaned:
                return cleaned
    return ""


def resolve_config(config: dict[str, str] | None = None) -> dict[str, str]:
    """Merge session token override into config when the user pastes it in the UI."""
    cfg = dict(config or load_config())
    manual_token = get_manual_twilio_token()
    if manual_token and is_valid_env_value(manual_token):
        cfg["TWILIO_AUTH_TOKEN"] = manual_token
    return cfg


def env_status_label(value: str) -> str:
    """Return a safe UI label for whether an env value is set."""
    if not value:
        return "missing"
    if not is_valid_env_value(value):
        return "placeholder"
    return "set"


def is_smtp_configured(config: dict[str, str] | None = None) -> bool:
    cfg = config or load_config()
    return all(
        is_valid_env_value(cfg.get(key, ""))
        for key in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM")
    )


def smtp_use_ssl(config: dict[str, str]) -> bool:
    """True when SMTP should connect with implicit SSL (SMTP_SSL port, e.g. 465)."""
    raw = (config.get("SMTP_SSL") or "true").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    try:
        return int(config.get("SMTP_PORT") or "465") == 465
    except ValueError:
        return True


def is_twilio_credentials_ok(config: dict[str, str] | None = None) -> bool:
    cfg = resolve_config(config)
    sid = cfg.get("TWILIO_ACCOUNT_SID", "")
    token = cfg.get("TWILIO_AUTH_TOKEN", "")
    return (
        is_valid_env_value(sid)
        and is_valid_env_value(token)
        and sid.startswith("AC")
        and len(token) >= 20
    )


def is_twilio_configured(config: dict[str, str] | None = None) -> bool:
    cfg = resolve_config(config)
    whatsapp_from = cfg.get("TWILIO_WHATSAPP_FROM", "")
    return is_twilio_credentials_ok(cfg) and is_valid_env_value(whatsapp_from)


def get_twilio_voice_from(config: dict[str, str]) -> str:
    """Twilio number with Voice capability; TWILIO_VOICE_FROM or TWILIO_SMS_FROM."""
    voice = config.get("TWILIO_VOICE_FROM", "").strip()
    if is_valid_env_value(voice) and voice.startswith("+"):
        return voice
    sms = config.get("TWILIO_SMS_FROM", "").strip()
    if is_valid_env_value(sms) and sms.startswith("+"):
        return sms
    return ""


def is_voice_configured(config: dict[str, str] | None = None) -> bool:
    cfg = resolve_config(config)
    return is_twilio_credentials_ok(cfg) and bool(get_twilio_voice_from(cfg))


def is_sms_configured(config: dict[str, str] | None = None) -> bool:
    cfg = resolve_config(config)
    sms_from = cfg.get("TWILIO_SMS_FROM", "")
    return (
        is_twilio_credentials_ok(cfg)
        and is_valid_env_value(sms_from)
        and sms_from.startswith("+")
    )


def get_api_key() -> str:
    """Return API key from .env file or Streamlit secrets (not stale os.environ)."""
    return load_config()["GEMINI_API_KEY"]


def get_gemini_model() -> str:
    return load_config()["GEMINI_MODEL"]


APP_VERSION = "2.1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = str(APP_DIR / "renewal_guardian_v2.db")

PAYMENT_FREQUENCIES = ["monthly", "quarterly", "semester", "annual"]

TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "product_title": "Important non-frequent issues reminder",
        "app_name": "Example: insurance renewal guardian",
        "app_version": "Version 2",
        "tagline": "Never lose your coverage",
        "language_label": "Language",
        "sidebar_steps_title": "How it works",
        "step_1": "Upload your policy PDF",
        "step_2": "Review renewal date and payment type",
        "step_3": "Set up reminders (AI suggests a schedule)",
        "step_4": "Upload payment proof to stop reminders",
        "upload_current": "Upload your policy",
        "upload_current_help": "Upload the PDF for your current Gastos Médicos Mayores policy",
        "upload_file_hint": "200MB per file • PDF",
        "analyze_btn": "Analyze my policy",
        "policy_mgmt_title": "Policy Management",
        "policy_mgmt_new": "Create New Policy",
        "policy_mgmt_active": "Active Policies",
        "policy_mgmt_past": "Past Policies",
        "section_policy_holder": "Policy Holder & Insurer",
        "section_renewal": "Renewal Date & Premium",
        "section_payment_type": "Payment Type",
        "section_insurance_type": "Insurance Type",
        "section_coverage": "Further coverage details",
        "section_questions": "Smart Questions to Ask Your Agent",
        "section_questions_desc": "Use these questions to make sure you fully understand your policy before it renews.",
        "section_questions_view": "View smart questions",
        "section_coverage_questions_expander": "Coverage details & smart questions",
        "section_risk": "Risk Alert",
        "section_ai_reminder": "AI Reminder Suggestion",
        "reminder_title": "Set up reminders",
        "reminder_tab_email": "Email",
        "reminder_tab_whatsapp": "WhatsApp",
        "channel_configuring_email": "You are configuring Email reminders",
        "channel_configuring_whatsapp": "You are configuring WhatsApp reminders",
        "reminder_timing_group": "Reminder Timing",
        "reminder_recipients_group": "Recipients",
        "whatsapp_primary": "Primary WhatsApp number",
        "whatsapp_secondary": "Secondary WhatsApp number (optional)",
        "field_required": "Required",
        "field_optional": "Optional",
        "reminder_plan_title": "Your reminder plan",
        "reminder_plan_desc": "Frequency increases as the due date approaches.",
        "reminder_plan_view": "View schedule",
        "reminder_plan_scroll_hint": "Scroll inside the table to see the full list.",
        "reminder_tier_normal": "Normal (once daily)",
        "reminder_tier_frequent": "Frequent ({count}x daily)",
        "email_primary": "Primary email",
        "email_secondary": "Secondary email (optional)",
        "reminder_days": "Daily reminder before due date (days)",
        "reminder_time": "Preferred daily time",
        "reminder_frequent_days": "Urgent reminders (more than once daily)",
        "reminder_frequent_days_help": "Must be lower than the daily reminder before due date value above.",
        "reminder_daily_frequency": "Daily frequency",
        "whatsapp_phone": "Mobile phone — voice / SMS / WhatsApp",
        "mobile_phone_verified_hint": "Copy your number **exactly** from Twilio (Verified Caller IDs or WhatsApp sandbox). The app never adds or removes digits.",
        "voice_config_title": "Voice call (Twilio) configuration",
        "voice_config_ok": "Voice is configured — no A2P 10DLC registration needed",
        "voice_config_missing": "Add a Twilio number with Voice enabled to .env:",
        "voice_config_trial_note": "Trial accounts: verify the recipient under Verified Caller IDs. Voice does not require A2P 10DLC (unlike US SMS).",
        "send_voice_now": "Call me now (voice demo)",
        "voice_sent": "Voice call started",
        "voice_sent_detail": "Twilio status: **{status}** · SID: `{sid}`",
        "voice_sent_note": "Your phone should ring shortly.",
        "voice_failed": "Could not start voice call",
        "voice_failed_detail": "Twilio error: {detail}",
        "voice_not_configured": "Voice not configured — add TWILIO_VOICE_FROM to .env",
        "sms_config_title": "SMS (Twilio) configuration",
        "sms_config_ok": "SMS is configured in .env",
        "sms_config_missing": "Add these to your .env file to enable SMS:",
        "sms_config_trial_note": "US numbers: SMS requires A2P 10DLC registration (see your number capabilities). Use Voice for demos instead.",
        "send_sms_now": "Send SMS reminder now",
        "sms_sent": "SMS reminder sent successfully",
        "sms_sent_detail": "Twilio accepted the message. Status: **{status}** · SID: `{sid}`",
        "sms_sent_note": "Check Twilio Console → Monitor → Logs → Messaging if it does not arrive.",
        "sms_failed": "Could not send SMS. Check your number and Twilio SMS settings",
        "sms_failed_detail": "Twilio error: {detail}",
        "twilio_trial_length_hint": "Twilio trial accounts only deliver single-segment messages (about 160 characters). This app now sends a shorter reminder for SMS and WhatsApp.",
        "sms_not_configured": "SMS not configured — add TWILIO_SMS_FROM to .env (see below).",
        "sms_no_phone": "Enter a mobile phone number first",
        "whatsapp_config_title": "WhatsApp (Twilio) configuration",
        "whatsapp_config_ok": "Twilio is configured in .env",
        "whatsapp_config_ok_session": "Twilio ready — using token from text box ({count} chars)",
        "whatsapp_config_missing": "Add these to your .env file to enable WhatsApp:",
        "twilio_token_manual": "Twilio Auth Token (this session)",
        "twilio_token_manual_help": "Paste your Auth Token from Twilio Console. Overrides the .env value while this app is running. Set it once in the sidebar.",
        "twilio_token_manual_active": "Token in text box: {count} characters — Twilio voice, SMS & WhatsApp enabled",
        "twilio_token_sidebar_hint": "Open **Configuration & secrets** in the sidebar and paste your Auth Token there.",
        "messaging_config_sidebar_hint": "Open **Configuration & secrets** in the sidebar to set Twilio and email credentials.",
        "whatsapp_available_soon": "WhatsApp reminders not configured — add Twilio keys to .env",
        "send_email_now": "Send email reminder now",
        "send_whatsapp_now": "Send WhatsApp reminder now",
        "whatsapp_demo_open": "Open in WhatsApp (demo)",
        "whatsapp_demo_help": "Opens WhatsApp with the reminder pre-filled. Works on any phone. An active WhatsApp app instance running on the computer is needed.",
        "whatsapp_twilio_label": "Automatic send (Twilio)",
        "whatsapp_production_note": "For production: register a WhatsApp Business sender in Twilio so any customer phone receives messages without joining a sandbox.",
        "email_sent": "Email reminder sent successfully",
        "email_failed": "Could not send email. Check SMTP settings in .env",
        "email_not_configured": "Email sending not configured. Add SMTP settings to .env (see below).",
        "email_no_address": "Enter a primary email address first",
        "whatsapp_sent": "WhatsApp reminder sent successfully",
        "whatsapp_sent_detail": "Twilio accepted the message. Delivery status: **{status}** · SID: `{sid}`",
        "whatsapp_sent_note": "If it does not arrive, check Twilio Console → Monitor → Logs → Messaging. Sandbox users must send the join code first.",
        "whatsapp_failed": "Could not send WhatsApp reminder. Check your number and Twilio .env settings",
        "whatsapp_failed_detail": "Twilio error: {detail}",
        "twilio_sandbox_join": (
            "**Error 63015** — Twilio does not recognize this number as a sandbox participant.\n\n"
            "1. Open [WhatsApp Sandbox](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn) "
            "→ **Sandbox participants** — your number must appear **exactly** as sent below.\n"
            "2. Copy the number **exactly** from Sandbox participants — do not add or remove digits.\n"
            "3. Re-send `join your-code` if you joined more than **72 hours** ago.\n"
            "4. The phone that joined must be the **same** number typed in the app."
        ),
        "twilio_verified_mismatch": (
            "**Error 21219 — number format mismatch.** Twilio lists your phone in Verified Caller IDs, "
            "but the digits sent by the app differ from Verified Caller IDs.\n\n"
            "1. Open [Verified Caller IDs](https://console.twilio.com/us1/develop/phone-numbers/manage/verified)\n"
            "2. **Copy the number exactly** as shown there\n"
            "3. Paste it in the mobile phone field — voice/SMS use that format **literally** (no auto-fix)\n"
            "4. Sent to: `{sent}` — must match Verified Caller IDs character for character"
        ),
        "send_now_title": "Send reminders now",
        "send_now_desc": "Use these buttons to send a reminder immediately. Your full schedule is in the plan above and in the .ics calendar.",
        "when_sent_desc": (
            "You can send an Email or WhatsApp reminder when you click the buttons below. "
            "The .ics calendar triggers reminders at scheduled times. "
            "You can also simulate a specific date at the top of the page."
        ),
        "smtp_config_title": "Email (SMTP) configuration",
        "smtp_config_ok": "SMTP is configured in .env (SSL connection)",
        "smtp_config_missing": "Add these to your .env file to enable secure email sending:",
        "smtp_config_ssl_note": "Email is sent over SSL (port 465). Set SMTP_SSL=false and SMTP_PORT=587 only if your provider requires STARTTLS instead.",
        "env_var_set": "Set",
        "env_var_missing": "Missing",
        "env_var_placeholder": "Placeholder — replace with your real value",
        "twilio_token_hint": "TWILIO_AUTH_TOKEN must be your real token from Twilio Console, not [AuthToken].",
        "env_example_only": "Example only — edit guardian_secrets.env, not this box:",
        "env_edit_path": "Open guardian_secrets.env in Notepad, paste your token, save, then restart the app:",
        "download_ics": "Download full reminder calendar (.ics)",
        "payment_frequency_label": "Payment frequency",
        "freq_monthly": "Monthly",
        "freq_quarterly": "Quarterly",
        "freq_semester": "Semester (every 6 months)",
        "freq_annual": "Annual",
        "payment_proof_title": "Confirm payment",
        "payment_proof_desc": "Upload your payment receipt to stop all reminders for this period.",
        "payment_proof_upload": "Upload payment proof",
        "payment_proof_help": "PDF or image of your payment confirmation",
        "payment_confirm_btn": "Confirm payment and stop reminders",
        "payment_confirmed": "Payment confirmed. Reminders stopped for this period.",
        "reminders_active": "Reminders are active",
        "reminders_stopped": "Reminders stopped — payment confirmed",
        "history_title": "Analysis history",
        "history_col_policy": "Policy",
        "history_col_renewal": "Renewal status",
        "history_col_payment": "Payment status",
        "renewal_status_good": "Good",
        "renewal_status_renew": "Renew",
        "renewal_status_urgent": "Urgent renewal!!",
        "history_empty": "No past analyses yet. Upload a policy to get started.",
        "history_load_hint": "Open an entry to view details and manage reminders.",
        "err_pdf_unreadable": "We couldn't read this PDF. Try a clearer scan or copy-paste the text manually",
        "err_empty_pdf": "This file appears to be empty. Upload a PDF with visible text",
        "err_no_renewal": "Due date not found — check page 1 of your policy",
        "err_api_failure": "Analysis unavailable right now. Try again in a moment",
        "err_no_api_key": "Gemini API key not found. Save your key in .env, then restart the app.",
        "env_file_label": "Secrets file",
        "env_file_hint": "Use `{name}` in the project folder (not `.env`). Old `.env` is ignored.",
        "env_key_missing": "API key not found on disk. Save guardian_secrets.env (Ctrl+S), then restart.",
        "env_key_ok": "API key loaded",
        "env_disk_hint": "If your editor shows a key but this says missing, the file is not saved to disk yet.",
        "env_disk_stats": "On disk: {size} bytes · last saved {modified}",
        "env_value_chars": "{count} characters read from file",
        "env_encoding": "File encoding: {encoding}",
        "env_source_file": "loaded from .env file",
        "env_source_secrets": "loaded from Streamlit secrets",
        "env_source_missing": "not found on disk",
        "env_os_ignored": "Cached Windows environment variables are cleared when secrets file is loaded.",
        "env_model_on_disk": "Model on disk",
        "status_confirmed": "Renewal confirmed — you are on track",
        "status_pending": "Payment due soon — review your policy",
        "status_overdue": "Payment overdue or imminent — act now",
        "risk_within_30": "Your payment is due within 30 days. Contact your agent.",
        "risk_within_7": "Your payment is due within 7 days. Act immediately to avoid a lapse.",
        "risk_ok": "Your due date is more than 30 days away. You are in good standing.",
        "analysis_saved": "Analysis saved to your history",
        "premium_label": "Premium",
        "renewal_label": "Next due date",
        "next_payment_due_date_label": "Next payment due date",
        "next_renewal_date_label": "Next renewal date",
        "payment_alert_on_time": " Everything Good",
        "payment_alert_pay": "Due date approaching",
        "payment_alert_pay_now": "Pay Now !!",
        "not_found": "Not found",
        "restart_btn": "Restart app",
        "restart_done": "App restarted. Upload a new policy to begin.",
        "delete_btn_help": "Delete this analysis",
        "delete_confirm_title": "Delete analysis?",
        "delete_irreversible": "This action is irreversible. The analysis will be permanently removed.",
        "delete_yes": "Yes, delete",
        "delete_no": "No, keep it",
        "delete_success": "Analysis deleted.",
        "analysis_results_title": "Your policy analysis",
        "analysis_language_note": "Analysis text is in {lang}. Run analysis again to regenerate in your selected language.",
        "lang_name_es": "Spanish",
        "lang_name_en": "English",
        "plan_col_date": "Date",
        "plan_col_time": "Time",
        "plan_col_tier": "Frequency",
        "save_reminders": "Save reminder settings",
        "reminders_saved": "Reminder settings saved for this policy.",
        "sim_date_title": "Date simulation",
        "sim_date_label": "Simulation date",
        "sim_date_today": "Today",
        "renewal_alert_days_label": "Days to renewal alert",
        "renewal_red_flag_days_label": "Days to red flag renewal",
        "renewal_no_dates": "No dates available",
        "config_renewal_alerts_title": "Config renewal alerts",
        "config_global_contacts_title": "Default contacts",
        "global_contacts_saved": "Default email and WhatsApp saved.",
        "save_btn": "Save",
        "save_policy_required": "Save the policy first before storing reminder settings.",
        "sim_date_btn": "Simulate",
        "sim_date_help": "Pick a date and click Simulate to preview risk for all policies and send reminders due that day.",
        "sim_date_active": "Simulating **{date}**",
        "sim_dispatch_title": "Simulation results",
        "sim_no_alerts": "No reminders scheduled for this date.",
        "sim_no_contact": "Reminder due but no email or phone saved — configure contacts in the policy record.",
        "sim_no_records": "No saved policies to simulate.",
        "sim_record_due": "**{holder}** — {count} reminder(s) sent ({channels})",
        "sim_record_not_due": "{holder} — no reminder on this date",
        "sim_record_inactive": "{holder} — reminders stopped (payment confirmed)",
        "sim_record_no_renewal": "{holder} — no payment date",
        "sim_dispatch_done": "Processed {records} policy/policies for {date}. Sent {count} alert(s).",
        "sim_due_now": "Due on simulated date",
        "sidebar_config_title": "Configuration & secrets",
        "section_policy_id": "Policy number",
        "duplicate_title": "Same policy detected",
        "duplicate_message": "This policy number already exists ({count} version(s) in history).",
        "duplicate_keep_both": "Keep both versions",
        "duplicate_want_replace": "Replace previous version",
        "duplicate_delete_manual": "To replace the previous version, delete it manually from History below, then upload and analyze this PDF again.",
        "duplicate_saved_both": "Saved as version #{version}.",
        "duplicate_not_saved": "This analysis is not saved yet — choose an option above.",
        "duplicate_id_label": "Policy ID detected",
        "version_label": "Upload version",
        "payment_schedule_title": "Payment schedule",
        "extension_allowance_label": "Extension allowance (days)",
        "extension_allowance_help": "Payments after the due date within this many days still count as valid.",
        "extension_allowance_saved": "Extension allowance saved.",
        "payment_item_summary": "Payment {n}/{total} — {date} · {status}",
        "payment_days_left": "{days} days until due",
        "payment_days_overdue": "{days} days overdue",
        "payment_due_today": "Due today",
        "payment_detail_due": "Due date",
        "payment_detail_status": "Status",
        "payment_status_ok": "On track (> 30 days)",
        "payment_status_warning": "Approaching (8–30 days)",
        "payment_status_critical": "Urgent (7 days or less)",
        "payment_status_paid": "Paid",
        "payment_upload_title": "Upload payment receipt",
        "payment_upload_help": "PDF or image of the payment receipt for this specific payment",
        "payment_verify_btn": "Verify payment",
        "payment_verifying": "Reading receipt...",
        "payment_extract_error": "Could not read the payment receipt. Try a clearer file.",
        "payment_missing_title": "Some required information was not found on the receipt",
        "payment_missing_hint": "Enter the missing details below, or upload a different receipt.",
        "payment_manual_continue": "Continue with entered details",
        "payment_try_again": "Upload a different receipt",
        "payment_confirmed_badge": "PAID — confirmed {date}",
        "payment_confirmed_auto": "All details matched — marked as paid automatically.",
        "payment_confirmed_override": "Marked as paid (manually validated despite mismatches).",
        "payment_mismatch_title": "Some details on the receipt don't match this payment",
        "payment_field_amount": "Amount",
        "payment_field_insurance_id": "Policy number",
        "payment_field_date": "Payment date",
        "payment_field_company": "Company",
        "payment_field_beneficiary": "Beneficiary",
        "payment_extracted_title": "Details read from receipt",
        "payment_expected_label": "Expected",
        "payment_found_label": "Found",
        "payment_match_ok": "Match",
        "payment_match_fail": "Mismatch",
        "payment_mismatch_reupload_btn": "Upload a different receipt",
        "payment_mismatch_override_btn": "Validate anyway",
        "payment_mismatch_override_warning": "If you confirm, this payment will be marked as paid and no further reminders will be sent for it — even though some details did not match.",
        "payment_mismatch_confirm_btn": "Yes, mark as paid",
        "upload_error_ok": "OK",
        "upload_error_title": "Could not read file",
        "upload_error_detail_label": "Reason",
        "upload_missing_title": "Some required information was not found in the PDF",
        "upload_missing_hint": "Enter the missing details below, or upload a different file.",
        "upload_manual_continue": "Continue with entered details",
        "upload_try_again": "Upload a different file",
        "field_plan_id": "Plan ID / policy number",
        "field_payment_frequency": "Payment frequency",
        "field_total_amount": "Total plan amount",
        "field_insurance_type": "Type of insurance",
        "field_insurance_company": "Insurance company",
        "field_payment_frequency_help": "annual, semester, quarterly, or monthly",
        "upload_missing_field_item": "• {field}",
        "plan_start_date_label": "Plan start date",
        "plan_end_date_label": "Plan end date",
        "renewal_premium_total_label": "Renewal premium (total)",
        "premium_per_term_label": "Premium per term",
        "status_on_time": "ON TIME",
        "status_warning_approaching": "WARNING, DUE DATE APPROACHING",
        "status_do_it_now": "DO IT NOW",
    },
    "es": {
        "product_title": "Recordatorio de temas importantes no frecuentes",
        "app_name": "Ejemplo: guardián de renovación de seguros",
        "app_version": "Versión 2",
        "tagline": "Nunca pierdas tu cobertura",
        "language_label": "Idioma",
        "sidebar_steps_title": "Cómo funciona",
        "step_1": "Sube el PDF de tu póliza",
        "step_2": "Revisa la fecha de pago y tipo de plan",
        "step_3": "Configura recordatorios (la IA sugiere un horario)",
        "step_4": "Sube tu comprobante de pago para detener recordatorios",
        "upload_current": "Sube tu póliza",
        "upload_current_help": "Sube el PDF de tu póliza actual de Gastos Médicos Mayores",
        "upload_file_hint": "200MB por archivo • PDF",
        "analyze_btn": "Analizar mi póliza",
        "policy_mgmt_title": "Gestión de Pólizas",
        "policy_mgmt_new": "Crear Nueva Póliza",
        "policy_mgmt_active": "Pólizas Activas",
        "policy_mgmt_past": "Pólizas Pasadas",
        "section_policy_holder": "Titular y Aseguradora",
        "section_renewal": "Fecha de Pago y Prima",
        "section_payment_type": "Tipo de Pago",
        "section_insurance_type": "Tipo de seguro",
        "section_coverage": "Detalles adicionales de cobertura",
        "section_questions": "Preguntas Inteligentes para tu Agente",
        "section_questions_desc": "Usa estas preguntas para asegurarte de entender bien tu póliza antes de que se renueve.",
        "section_questions_view": "Ver preguntas inteligentes",
        "section_coverage_questions_expander": "Detalles de cobertura y preguntas inteligentes",
        "section_risk": "Alerta de Riesgo",
        "section_ai_reminder": "Sugerencia de Recordatorios (IA)",
        "reminder_title": "Configurar recordatorios",
        "reminder_tab_email": "Correo",
        "reminder_tab_whatsapp": "WhatsApp",
        "channel_configuring_email": "Estás configurando los recordatorios por Correo",
        "channel_configuring_whatsapp": "Estás configurando los recordatorios por WhatsApp",
        "reminder_timing_group": "Horario de recordatorios",
        "reminder_recipients_group": "Destinatarios",
        "whatsapp_primary": "Número principal de WhatsApp",
        "whatsapp_secondary": "Número secundario de WhatsApp (opcional)",
        "field_required": "Requerido",
        "field_optional": "Opcional",
        "reminder_plan_title": "Tu plan de recordatorios",
        "reminder_plan_desc": "La frecuencia aumenta conforme se acerca la fecha de pago.",
        "reminder_plan_view": "Ver calendario",
        "reminder_plan_scroll_hint": "Desplázate dentro de la tabla para ver la lista completa.",
        "reminder_tier_normal": "Normal (1 vez al día)",
        "reminder_tier_frequent": "Frecuente ({count}x al día)",
        "email_primary": "Correo principal",
        "email_secondary": "Correo secundario (opcional)",
        "reminder_days": "Recordatorio diario antes del pago (días)",
        "reminder_time": "Hora diaria preferida",
        "reminder_frequent_days": "Recordatorios urgentes (más de una vez al día)",
        "reminder_frequent_days_help": "Debe ser menor que el recordatorio diario antes del pago de arriba.",
        "reminder_daily_frequency": "Frecuencia diaria",
        "whatsapp_phone": "Celular — voz / SMS / WhatsApp",
        "mobile_phone_verified_hint": "Copia tu número **exactamente** desde Twilio (Verified Caller IDs o sandbox WhatsApp). La app nunca agrega ni quita dígitos.",
        "voice_config_title": "Llamada de voz (Twilio)",
        "voice_config_ok": "Voz configurada — no requiere registro A2P 10DLC",
        "voice_config_missing": "Agrega un número Twilio con Voz habilitada en .env:",
        "voice_config_trial_note": "Cuentas trial: verifica el destinatario en Verified Caller IDs. Voz no requiere A2P 10DLC (a diferencia de SMS en EE.UU.).",
        "send_voice_now": "Llámame ahora (demo voz)",
        "voice_sent": "Llamada iniciada",
        "voice_sent_detail": "Estado Twilio: **{status}** · SID: `{sid}`",
        "voice_sent_note": "Tu teléfono debería sonar en breve.",
        "voice_failed": "No se pudo iniciar la llamada",
        "voice_failed_detail": "Error de Twilio: {detail}",
        "voice_not_configured": "Voz no configurada — agrega TWILIO_VOICE_FROM en .env",
        "sms_config_title": "Configuración de SMS (Twilio)",
        "sms_config_ok": "SMS está configurado en .env",
        "sms_config_missing": "Agrega esto a tu .env para activar SMS:",
        "sms_config_trial_note": "Números EE.UU.: SMS requiere registro A2P 10DLC. Usa Voz para demos.",
        "send_sms_now": "Enviar recordatorio por SMS ahora",
        "sms_sent": "Recordatorio por SMS enviado correctamente",
        "sms_sent_detail": "Twilio aceptó el mensaje. Estado: **{status}** · SID: `{sid}`",
        "sms_sent_note": "Revisa Twilio Console → Monitor → Logs → Messaging si no llega.",
        "sms_failed": "No se pudo enviar SMS. Verifica tu número y la configuración SMS de Twilio",
        "sms_failed_detail": "Error de Twilio: {detail}",
        "twilio_trial_length_hint": "Las cuentas trial de Twilio solo entregan mensajes de un segmento (unos 160 caracteres). La app ahora envia un recordatorio mas corto por SMS y WhatsApp.",
        "sms_not_configured": "SMS no configurado — agrega TWILIO_SMS_FROM en .env (ver abajo).",
        "sms_no_phone": "Ingresa un número de celular primero",
        "whatsapp_config_title": "Configuración de WhatsApp (Twilio)",
        "whatsapp_config_ok": "Twilio está configurado en .env",
        "whatsapp_config_ok_session": "Twilio listo — token del cuadro de texto ({count} chars)",
        "whatsapp_config_missing": "Agrega esto a tu .env para activar WhatsApp:",
        "twilio_token_manual": "Twilio Auth Token (esta sesión)",
        "twilio_token_manual_help": "Pega tu Auth Token de Twilio Console. Reemplaza el valor de .env mientras la app esté abierta. Configúralo una vez en la barra lateral.",
        "twilio_token_manual_active": "Token en cuadro: {count} caracteres — Twilio voz, SMS y WhatsApp activos",
        "twilio_token_sidebar_hint": "Abre **Configuración y secretos** en la barra lateral y pega tu Auth Token ahí.",
        "messaging_config_sidebar_hint": "Abre **Configuración y secretos** en la barra lateral para configurar Twilio y correo.",
        "whatsapp_available_soon": "WhatsApp no configurado — agrega las claves de Twilio en .env",
        "send_email_now": "Enviar recordatorio por correo ahora",
        "send_whatsapp_now": "Enviar recordatorio por WhatsApp ahora",
        "whatsapp_demo_open": "Abrir en WhatsApp (demo)",
        "whatsapp_demo_help": "Abre WhatsApp con el recordatorio prellenado. Funciona en cualquier teléfono. Se necesita una instancia activa de WhatsApp en la computadora.",
        "whatsapp_twilio_label": "Envío automático (Twilio)",
        "whatsapp_production_note": "Para producción: registra un número WhatsApp Business en Twilio para que cualquier cliente reciba mensajes sin unirse al sandbox.",
        "email_sent": "Recordatorio por correo enviado correctamente",
        "email_failed": "No se pudo enviar el correo. Revisa la configuración SMTP en .env",
        "email_not_configured": "Correo no configurado. Agrega SMTP en .env (ver abajo).",
        "email_no_address": "Ingresa un correo principal primero",
        "whatsapp_sent": "Recordatorio por WhatsApp enviado correctamente",
        "whatsapp_sent_detail": "Twilio aceptó el mensaje. Estado: **{status}** · SID: `{sid}`",
        "whatsapp_sent_note": "Si no llega, revisa Twilio Console → Monitor → Logs → Messaging. En sandbox debes enviar el código join primero.",
        "whatsapp_failed": "No se pudo enviar por WhatsApp. Verifica tu número y Twilio en .env",
        "whatsapp_failed_detail": "Error de Twilio: {detail}",
        "twilio_sandbox_join": (
            "**Error 63015** — Twilio no reconoce este número como participante del sandbox.\n\n"
            "1. Abre [WhatsApp Sandbox](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn) "
            "→ **Sandbox participants** — tu número debe aparecer **igual** al enviado abajo.\n"
            "2. Copia el número **exactamente** de Sandbox participants — no agregues ni quites dígitos.\n"
            "3. Reenvía `join tu-código` si te uniste hace más de **72 horas**.\n"
            "4. El teléfono que se unió debe ser el **mismo** que escribes en la app."
        ),
        "twilio_verified_mismatch": (
            "**Error 21219 — formato distinto.** Twilio muestra tu teléfono en Verified Caller IDs, "
            "pero los dígitos enviados difieren de Verified Caller IDs.\n\n"
            "1. Abre [Verified Caller IDs](https://console.twilio.com/us1/develop/phone-numbers/manage/verified)\n"
            "2. **Copia el número exactamente** como aparece\n"
            "3. Pégalo en el campo de celular — voz/SMS usan ese formato **literal** (sin auto-corrección)\n"
            "4. Enviado a: `{sent}` — debe coincidir carácter por carácter"
        ),
        "send_now_title": "Enviar recordatorios ahora",
        "send_now_desc": "Usa estos botones para enviar un recordatorio de inmediato. El plan completo está arriba y en el calendario .ics.",
        "when_sent_desc": (
            "Puedes enviar un recordatorio por correo o WhatsApp al hacer clic en los botones de abajo. "
            "El calendario .ics activa recordatorios en los horarios programados. "
            "También puedes simular una fecha específica en la parte superior de la página."
        ),
        "smtp_config_title": "Configuración de correo (SMTP)",
        "smtp_config_ok": "SMTP está configurado en .env (conexión SSL)",
        "smtp_config_missing": "Agrega esto a tu .env para activar el envío seguro de correos:",
        "smtp_config_ssl_note": "El correo se envía por SSL (puerto 465). Usa SMTP_SSL=false y SMTP_PORT=587 solo si tu proveedor requiere STARTTLS.",
        "env_var_set": "Configurado",
        "env_var_missing": "Falta",
        "env_var_placeholder": "Placeholder — reemplaza con tu valor real",
        "twilio_token_hint": "TWILIO_AUTH_TOKEN debe ser tu token real de Twilio Console, no [AuthToken].",
        "env_example_only": "Solo un ejemplo — edita guardian_secrets.env, no este cuadro:",
        "env_edit_path": "Abre guardian_secrets.env en Notepad, pega tu token, guarda y reinicia la app:",
        "download_ics": "Descargar calendario completo (.ics)",
        "payment_frequency_label": "Frecuencia de pago",
        "freq_monthly": "Mensual",
        "freq_quarterly": "Trimestral",
        "freq_semester": "Semestral",
        "freq_annual": "Anual",
        "payment_proof_title": "Confirmar pago",
        "payment_proof_desc": "Sube tu comprobante de pago para detener todos los recordatorios de este periodo.",
        "payment_proof_upload": "Subir comprobante de pago",
        "payment_proof_help": "PDF o imagen de tu confirmación de pago",
        "payment_confirm_btn": "Confirmar pago y detener recordatorios",
        "payment_confirmed": "Pago confirmado. Recordatorios detenidos para este periodo.",
        "reminders_active": "Recordatorios activos",
        "reminders_stopped": "Recordatorios detenidos — pago confirmado",
        "history_title": "Historial de análisis",
        "history_col_policy": "Póliza",
        "history_col_renewal": "Estado de renovación",
        "history_col_payment": "Estado de pago",
        "renewal_status_good": "Al día",
        "renewal_status_renew": "Renovar",
        "renewal_status_urgent": "Urgente",
        "history_empty": "Aún no hay análisis previos. Sube una póliza para comenzar.",
        "history_load_hint": "Abre una entrada para ver detalles y gestionar recordatorios.",
        "err_pdf_unreadable": "No pudimos leer este PDF. Intenta con un escaneo más claro",
        "err_empty_pdf": "Este archivo parece estar vacío. Sube un PDF con texto visible",
        "err_no_renewal": "No se encontró la fecha de pago — revisa la página 1 de tu póliza",
        "err_api_failure": "Análisis no disponible en este momento. Intenta de nuevo",
        "err_no_api_key": "Clave de API no encontrada. Guarda tu clave en .env y reinicia la app.",
        "env_file_label": "Archivo de secretos",
        "env_file_hint": "Usa `{name}` en la carpeta del proyecto (no `.env`). El `.env` antiguo se ignora.",
        "env_key_missing": "Clave no encontrada en disco. Guarda guardian_secrets.env (Ctrl+S) y reinicia.",
        "env_key_ok": "Clave de API cargada",
        "env_disk_hint": "Si tu editor muestra la clave pero aquí dice que falta, el archivo aún no está guardado en disco.",
        "env_disk_stats": "En disco: {size} bytes · guardado {modified}",
        "env_value_chars": "{count} caracteres leídos del archivo",
        "env_encoding": "Codificación: {encoding}",
        "env_source_file": "leído del archivo .env",
        "env_source_secrets": "leído de Streamlit secrets",
        "env_source_missing": "no encontrado en disco",
        "env_os_ignored": "Variables de Windows en caché se borran al cargar el archivo de secretos.",
        "env_model_on_disk": "Modelo en disco",
        "status_confirmed": "Renovación al día",
        "status_pending": "Pago próximo — revisa tu póliza",
        "status_overdue": "Pago vencido o inminente — actúa ahora",
        "risk_within_30": "Tu pago vence en menos de 30 días. Contacta a tu agente.",
        "risk_within_7": "Tu pago vence en menos de 7 días. Actúa de inmediato.",
        "risk_ok": "Tu fecha de pago está a más de 30 días. Estás al corriente.",
        "analysis_saved": "Análisis guardado en tu historial",
        "premium_label": "Prima",
        "renewal_label": "Próxima fecha de pago",
        "next_payment_due_date_label": "Próxima fecha de pago",
        "next_renewal_date_label": "Próxima fecha de renovación",
        "payment_alert_on_time": "Todo bien",
        "payment_alert_pay": "Se acerca el pago",
        "payment_alert_pay_now": "Paga ahora !!",
        "not_found": "No encontrado",
        "restart_btn": "Reiniciar aplicación",
        "restart_done": "Aplicación reiniciada. Sube una nueva póliza para comenzar.",
        "delete_btn_help": "Eliminar este análisis",
        "delete_confirm_title": "¿Eliminar análisis?",
        "delete_irreversible": "Esta acción es irreversible. El análisis se eliminará permanentemente.",
        "delete_yes": "Sí, eliminar",
        "delete_no": "No, conservar",
        "delete_success": "Análisis eliminado.",
        "analysis_results_title": "Tu análisis de póliza",
        "analysis_language_note": "El texto del análisis está en {lang}. Vuelve a analizar para regenerarlo.",
        "lang_name_es": "español",
        "lang_name_en": "inglés",
        "plan_col_date": "Fecha",
        "plan_col_time": "Hora",
        "plan_col_tier": "Frecuencia",
        "save_reminders": "Guardar configuración de recordatorios",
        "reminders_saved": "Configuración de recordatorios guardada para esta póliza.",
        "sim_date_title": "Simulación de fecha",
        "sim_date_label": "Fecha simulada",
        "sim_date_today": "Hoy",
        "renewal_alert_days_label": "Días para alerta de renovación",
        "renewal_red_flag_days_label": "Días para alerta roja de renovación",
        "renewal_no_dates": "Sin fechas disponibles",
        "config_renewal_alerts_title": "Configurar alertas de renovación",
        "config_global_contacts_title": "Contactos predeterminados",
        "global_contacts_saved": "Correo y WhatsApp predeterminados guardados.",
        "save_btn": "Guardar",
        "save_policy_required": "Guarda la póliza primero antes de almacenar los recordatorios.",
        "sim_date_btn": "Simular",
        "sim_date_help": "Elige una fecha y pulsa Simular para ver el riesgo de todas las pólizas y enviar recordatorios de ese día.",
        "sim_date_active": "Simulando **{date}**",
        "sim_dispatch_title": "Resultados de la simulación",
        "sim_no_alerts": "No hay recordatorios programados para esta fecha.",
        "sim_no_contact": "Hay recordatorio este día pero no hay correo ni teléfono guardados — configúralos en la póliza.",
        "sim_no_records": "No hay pólizas guardadas para simular.",
        "sim_record_due": "**{holder}** — {count} recordatorio(s) enviado(s) ({channels})",
        "sim_record_not_due": "{holder} — sin recordatorio en esta fecha",
        "sim_record_inactive": "{holder} — recordatorios detenidos (pago confirmado)",
        "sim_record_no_renewal": "{holder} — sin fecha de pago",
        "sim_dispatch_done": "Se procesaron {records} póliza(s) para {date}. Se enviaron {count} alerta(s).",
        "sim_due_now": "Vence en la fecha simulada",
        "sidebar_config_title": "Configuración y secretos",
        "section_policy_id": "Número de póliza",
        "duplicate_title": "Misma póliza detectada",
        "duplicate_message": "Este número de póliza ya existe ({count} versión(es) en el historial).",
        "duplicate_keep_both": "Conservar ambas versiones",
        "duplicate_want_replace": "Reemplazar versión anterior",
        "duplicate_delete_manual": "Para reemplazar la versión anterior, elimínala manualmente en el Historial abajo y vuelve a subir y analizar este PDF.",
        "duplicate_saved_both": "Guardado como versión #{version}.",
        "duplicate_not_saved": "Este análisis aún no se guarda — elige una opción arriba.",
        "duplicate_id_label": "ID de póliza detectado",
        "version_label": "Versión de carga",
        "payment_schedule_title": "Calendario de pagos",
        "extension_allowance_label": "Tolerancia de prórroga (días)",
        "extension_allowance_help": "Los pagos después del vencimiento dentro de estos días aún se consideran válidos.",
        "extension_allowance_saved": "Tolerancia de prórroga guardada.",
        "payment_item_summary": "Pago {n}/{total} — {date} · {status}",
        "payment_days_left": "{days} días para el vencimiento",
        "payment_days_overdue": "{days} días de atraso",
        "payment_due_today": "Vence hoy",
        "payment_detail_due": "Fecha de pago",
        "payment_detail_status": "Estado",
        "payment_status_ok": "Al día (> 30 días)",
        "payment_status_warning": "Próximo (8–30 días)",
        "payment_status_critical": "Urgente (7 días o menos)",
        "payment_status_paid": "Pagado",
        "payment_upload_title": "Subir comprobante de pago",
        "payment_upload_help": "PDF o imagen del comprobante de pago para este pago específico",
        "payment_verify_btn": "Verificar pago",
        "payment_verifying": "Leyendo comprobante...",
        "payment_extract_error": "No se pudo leer el comprobante de pago. Intenta con un archivo más claro.",
        "payment_missing_title": "No se encontró información requerida en el comprobante",
        "payment_missing_hint": "Ingresa los datos faltantes abajo, o sube otro comprobante.",
        "payment_manual_continue": "Continuar con los datos ingresados",
        "payment_try_again": "Subir otro comprobante",
        "payment_confirmed_badge": "PAGADO — confirmado {date}",
        "payment_confirmed_auto": "Todos los datos coinciden — marcado como pagado automáticamente.",
        "payment_confirmed_override": "Marcado como pagado (validado manualmente a pesar de las diferencias).",
        "payment_mismatch_title": "Algunos datos del comprobante no coinciden con este pago",
        "payment_field_amount": "Monto",
        "payment_field_insurance_id": "Número de póliza",
        "payment_field_date": "Fecha de pago",
        "payment_field_company": "Compañía",
        "payment_field_beneficiary": "Beneficiario",
        "payment_extracted_title": "Datos leídos del comprobante",
        "payment_expected_label": "Esperado",
        "payment_found_label": "Encontrado",
        "payment_match_ok": "Coincide",
        "payment_match_fail": "No coincide",
        "payment_mismatch_reupload_btn": "Subir otro comprobante",
        "payment_mismatch_override_btn": "Validar de todas formas",
        "payment_mismatch_override_warning": "Si confirmas, este pago se marcará como pagado y no se enviarán más recordatorios para él — aunque algunos datos no coincidieron.",
        "payment_mismatch_confirm_btn": "Sí, marcar como pagado",
        "upload_error_ok": "OK",
        "upload_error_title": "No se pudo leer el archivo",
        "upload_error_detail_label": "Motivo",
        "upload_missing_title": "No se encontró información requerida en el PDF",
        "upload_missing_hint": "Ingresa los datos faltantes abajo, o sube otro archivo.",
        "upload_manual_continue": "Continuar con los datos ingresados",
        "upload_try_again": "Subir otro archivo",
        "field_plan_id": "ID de póliza / número de plan",
        "field_payment_frequency": "Frecuencia de pago",
        "field_total_amount": "Monto total del plan",
        "field_insurance_type": "Tipo de seguro",
        "field_insurance_company": "Compañía aseguradora",
        "field_payment_frequency_help": "anual, semestral, trimestral o mensual",
        "upload_missing_field_item": "• {field}",
        "plan_start_date_label": "Fecha de inicio del plan",
        "plan_end_date_label": "Fecha de fin del plan",
        "renewal_premium_total_label": "Prima de renovación (total)",
        "premium_per_term_label": "Prima por periodo",
        "status_on_time": "AL DÍA",
        "status_warning_approaching": "ADVERTENCIA, FECHA DE PAGO SE ACERCA",
        "status_do_it_now": "HAZLO AHORA",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("language", "en")
    return TEXTS.get(lang, TEXTS["es"]).get(key, key)


def freq_label(freq: str) -> str:
    return t(f"freq_{freq}")


def tier_label(tier: str, count: int = 1) -> str:
    return t(f"reminder_tier_{tier}").format(count=count)


def get_reference_date() -> date:
    return st.session_state.get("reference_date", date.today())


def extract_contacts_from_row(row: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    phone = ""
    phone_secondary = ""
    settings: dict[str, Any] = {}
    if row.get("reminder_settings_json"):
        try:
            settings = json.loads(row["reminder_settings_json"])
        except json.JSONDecodeError:
            settings = {}
        phone = (settings.get("whatsapp_phone") or "").strip()
        phone_secondary = (settings.get("whatsapp_secondary") or "").strip()
        wa_channel = settings.get("whatsapp") or {}
        if isinstance(wa_channel, dict):
            phone = (wa_channel.get("primary") or phone).strip()
            phone_secondary = (wa_channel.get("secondary") or phone_secondary).strip()
    return {
        "email_primary": (row.get("email_primary") or context.get("email_primary") or "").strip(),
        "email_secondary": (row.get("email_secondary") or context.get("email_secondary") or "").strip(),
        "phone": phone,
        "phone_secondary": phone_secondary,
    }


def load_reminder_settings_blob(row: dict[str, Any] | None, context: dict[str, Any]) -> dict[str, Any]:
    if row and row.get("reminder_settings_json"):
        try:
            return json.loads(row["reminder_settings_json"])
        except json.JSONDecodeError:
            pass
    return {}


def parse_time_value(value: str) -> time:
    try:
        return datetime.strptime(str(value)[:5], "%H:%M").time()
    except ValueError:
        return datetime.strptime("09:00", "%H:%M").time()


def format_time_value(value: time | str) -> str:
    if isinstance(value, time):
        return value.strftime("%H:%M")
    return str(value)[:5] if value else "09:00"


def build_reminder_settings_blob(
    email_settings: dict[str, Any],
    whatsapp_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "email": email_settings,
        "whatsapp": whatsapp_settings,
        "payment_frequency": email_settings.get("payment_frequency", "annual"),
        "reminder_start_days": int(email_settings.get("reminder_start_days") or 30),
        "reminder_base_time": email_settings.get("reminder_base_time") or "09:00",
        "frequent_start_days": int(email_settings.get("frequent_start_days") or 7),
        "daily_frequency": int(email_settings.get("daily_frequency") or 2),
        "email_primary": (email_settings.get("primary") or "").strip(),
        "email_secondary": (email_settings.get("secondary") or "").strip(),
        "whatsapp_phone": (whatsapp_settings.get("primary") or "").strip(),
        "whatsapp_secondary": (whatsapp_settings.get("secondary") or "").strip(),
    }


def default_frequent_start_days(reminder_start_days: int) -> int:
    """Heuristic default for the 'more frequent' window, always < reminder_start_days."""
    reminder_start_days = max(1, int(reminder_start_days or 1))
    if reminder_start_days <= 1:
        return 1
    return max(1, min(7, reminder_start_days - 1))


def generate_daily_times(base_time: str, count: int) -> list[str]:
    """Spread `count` reminder times across the day, starting at base_time."""
    count = max(1, min(5, int(count or 1)))
    base = parse_time_value(base_time)
    if count == 1:
        return [format_time_value(base)]
    start_minutes = base.hour * 60 + base.minute
    end_minutes = max(start_minutes + 60, 22 * 60)
    step = (end_minutes - start_minutes) / (count - 1)
    times: list[str] = []
    for i in range(count):
        total = int(round(start_minutes + step * i)) % (24 * 60)
        times.append(f"{total // 60:02d}:{total % 60:02d}")
    return times


def reminder_key_prefix(analysis_id: int | None, *, scope: str = "active") -> str:
    if analysis_id:
        base = f"pol_{int(analysis_id)}_"
    else:
        base = "pol_draft_"
    if scope in {"active", "history"}:
        return f"{base}{scope}_"
    return base


def policy_payment_frequency(
    analysis: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    return normalize_payment_frequency(
        (context or {}).get("payment_frequency") or (analysis or {}).get("payment_frequency")
    )


def get_channel_reminder_config(
    channel: str,
    row: dict[str, Any] | None,
    context: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    settings = load_reminder_settings_blob(row, context)
    channel_data = settings.get(channel) if isinstance(settings.get(channel), dict) else {}
    legacy_days = int(
        context.get("reminder_start_days")
        or analysis.get("suggested_reminder_start_days", 30)
    )
    legacy_time = context.get("reminder_base_time") or analysis.get("suggested_reminder_time", "09:00")
    legacy_freq = normalize_payment_frequency(
        context.get("payment_frequency") or analysis.get("payment_frequency")
    )
    resolved_days = int(channel_data.get("reminder_start_days") or settings.get("reminder_start_days") or legacy_days)
    legacy_frequent_days = int(
        settings.get("frequent_start_days") or default_frequent_start_days(resolved_days)
    )
    legacy_daily_freq = int(settings.get("daily_frequency") or 2)
    contacts = extract_contacts_from_row(row or {}, context)
    global_contacts = load_global_contacts()
    if channel == "email":
        return {
            "payment_frequency": normalize_payment_frequency(
                channel_data.get("payment_frequency") or legacy_freq
            ),
            "reminder_start_days": resolved_days,
            "reminder_base_time": channel_data.get("reminder_base_time") or legacy_time,
            "frequent_start_days": int(channel_data.get("frequent_start_days") or legacy_frequent_days),
            "daily_frequency": int(channel_data.get("daily_frequency") or legacy_daily_freq),
            "primary": resolve_contact_primary("email", channel_data, contacts, global_contacts, settings),
            "secondary": resolve_contact_secondary("email", channel_data, contacts),
        }
    return {
        "payment_frequency": normalize_payment_frequency(
            channel_data.get("payment_frequency") or legacy_freq
        ),
        "reminder_start_days": resolved_days,
        "frequent_start_days": int(channel_data.get("frequent_start_days") or legacy_frequent_days),
        "daily_frequency": int(channel_data.get("daily_frequency") or legacy_daily_freq),
        "reminder_base_time": channel_data.get("reminder_base_time") or legacy_time,
        "primary": resolve_contact_primary("whatsapp", channel_data, contacts, global_contacts, settings),
        "secondary": resolve_contact_secondary("whatsapp", channel_data, contacts),
    }


def reminder_sync_token(row: dict[str, Any] | None, context: dict[str, Any]) -> str:
    global_contacts = load_global_contacts()
    global_part = f"{global_contacts['email_primary']}|{global_contacts['whatsapp_primary']}"
    if row:
        return "|".join([
            global_part,
            str(row.get("id", "")),
            row.get("reminder_settings_json") or "",
            row.get("email_primary") or "",
            row.get("email_secondary") or "",
            str(row.get("reminder_start_days") or ""),
            row.get("reminder_base_time") or "",
            row.get("payment_frequency") or "",
        ])
    return f"{global_part}|draft:{context.get('analysis_id', 'new')}"


def sync_reminder_fields_from_policy(
    key_prefix: str,
    row: dict[str, Any] | None,
    context: dict[str, Any],
    analysis: dict[str, Any],
) -> None:
    """Load saved reminder settings for this policy into session state."""
    sync_key = f"{key_prefix}reminder_sync_token"
    token = reminder_sync_token(row, context)
    if st.session_state.get(sync_key) == token:
        return

    email_cfg = get_channel_reminder_config("email", row, context, analysis)
    wa_cfg = get_channel_reminder_config("whatsapp", row, context, analysis)
    st.session_state[f"{key_prefix}email_payment_freq"] = email_cfg["payment_frequency"]
    st.session_state[f"{key_prefix}email_reminder_days"] = email_cfg["reminder_start_days"]
    st.session_state[f"{key_prefix}email_reminder_time"] = parse_time_value(email_cfg["reminder_base_time"])
    st.session_state[f"{key_prefix}email_primary"] = email_cfg["primary"]
    st.session_state[f"{key_prefix}email_secondary"] = email_cfg["secondary"]
    st.session_state[f"{key_prefix}wa_payment_freq"] = wa_cfg["payment_frequency"]
    st.session_state[f"{key_prefix}wa_reminder_days"] = wa_cfg["reminder_start_days"]
    st.session_state[f"{key_prefix}wa_reminder_time"] = parse_time_value(wa_cfg["reminder_base_time"])
    st.session_state[f"{key_prefix}whatsapp_phone"] = wa_cfg["primary"]
    st.session_state[f"{key_prefix}whatsapp_secondary"] = wa_cfg["secondary"]
    st.session_state[f"{key_prefix}payment_freq"] = email_cfg["payment_frequency"]
    st.session_state[f"{key_prefix}reminder_days"] = email_cfg["reminder_start_days"]
    st.session_state[f"{key_prefix}reminder_time"] = email_cfg["reminder_base_time"]
    # Unified schedule controls (shared by email + WhatsApp — one schedule per policy).
    st.session_state[f"{key_prefix}sched_days"] = email_cfg["reminder_start_days"]
    st.session_state[f"{key_prefix}sched_time"] = parse_time_value(email_cfg["reminder_base_time"])
    st.session_state[f"{key_prefix}sched_frequent_days"] = min(
        email_cfg["frequent_start_days"], max(1, email_cfg["reminder_start_days"] - 1)
    )
    st.session_state[f"{key_prefix}sched_daily_freq"] = email_cfg["daily_frequency"]
    st.session_state[f"{key_prefix}schedule_saved_signature"] = (
        int(email_cfg["reminder_start_days"]),
        min(email_cfg["frequent_start_days"], max(1, email_cfg["reminder_start_days"] - 1)),
        int(email_cfg["daily_frequency"]),
        email_cfg["reminder_base_time"],
    )
    st.session_state.pop(f"{key_prefix}schedule_save_flash", None)
    st.session_state[sync_key] = token


def seed_reminder_fields(
    key_prefix: str,
    row: dict[str, Any] | None,
    context: dict[str, Any],
    analysis: dict[str, Any],
) -> None:
    sync_reminder_fields_from_policy(key_prefix, row, context, analysis)


def read_schedule_form_values(key_prefix: str) -> dict[str, Any]:
    """Read the single, unified schedule-timing controls shared by all channels."""
    reminder_days = int(st.session_state.get(f"{key_prefix}sched_days", 30))
    frequent_days = int(st.session_state.get(f"{key_prefix}sched_frequent_days", default_frequent_start_days(reminder_days)))
    frequent_days = min(frequent_days, max(1, reminder_days - 1))
    return {
        "reminder_start_days": reminder_days,
        "reminder_base_time": format_time_value(
            st.session_state.get(f"{key_prefix}sched_time", parse_time_value("09:00"))
        ),
        "frequent_start_days": frequent_days,
        "daily_frequency": int(st.session_state.get(f"{key_prefix}sched_daily_freq", 2)),
    }


def read_channel_form_values(
    key_prefix: str,
    channel: str,
    *,
    analysis: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    freq = policy_payment_frequency(analysis, context)
    schedule = read_schedule_form_values(key_prefix)
    if channel == "email":
        return {
            "payment_frequency": freq,
            **schedule,
            "primary": st.session_state.get(f"{key_prefix}email_primary", ""),
            "secondary": st.session_state.get(f"{key_prefix}email_secondary", ""),
        }
    return {
        "payment_frequency": freq,
        **schedule,
        "primary": st.session_state.get(f"{key_prefix}whatsapp_phone", ""),
        "secondary": st.session_state.get(f"{key_prefix}whatsapp_secondary", ""),
    }


def render_compact_channel_recipients(
    channel: str,
    key_prefix: str,
    *,
    analysis_id: int | None,
    context: dict[str, Any],
) -> None:
    is_email = channel == "email"
    primary_key = f"{key_prefix}email_primary" if is_email else f"{key_prefix}whatsapp_phone"
    secondary_key = f"{key_prefix}email_secondary" if is_email else f"{key_prefix}whatsapp_secondary"
    primary_label = t("email_primary") if is_email else t("whatsapp_primary")
    secondary_label = t("email_secondary") if is_email else t("whatsapp_secondary")
    section_title = t("reminder_tab_email") if is_email else t("reminder_tab_whatsapp")

    st.markdown(f'<div class="reminder-group-title">{section_title}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="reminder-recipients-compact-marker"></div>', unsafe_allow_html=True)
        primary_col, secondary_col, save_col = st.columns(
            [2.2, 2.2, 0.85], gap="small", vertical_alignment="bottom"
        )
        with primary_col:
            st.text_input(primary_label, key=primary_key)
        with secondary_col:
            st.text_input(secondary_label, key=secondary_key)
        with save_col:
            st.markdown('<div class="save-btn-blue-marker"></div>', unsafe_allow_html=True)
            if st.button(t("save_btn"), key=f"{key_prefix}save_{channel}_contacts", width="stretch"):
                if analysis_id and context is not None:
                    if save_policy_reminder_settings(analysis_id, key_prefix, context):
                        st.success(t("reminders_saved"))
                else:
                    st.warning(t("save_policy_required"))
        if not is_email:
            st.caption(t("mobile_phone_verified_hint"))


def _schedule_timing_signature(key_prefix: str) -> tuple[int, int, int, str]:
    sched_time = st.session_state.get(f"{key_prefix}sched_time")
    if sched_time is not None and hasattr(sched_time, "strftime"):
        time_str = sched_time.strftime("%H:%M")
    else:
        time_str = str(sched_time or "09:00")
    return (
        int(st.session_state.get(f"{key_prefix}sched_days", 30)),
        int(st.session_state.get(f"{key_prefix}sched_frequent_days", 7)),
        int(st.session_state.get(f"{key_prefix}sched_daily_freq", 1)),
        time_str,
    )


def _schedule_timing_is_dirty(key_prefix: str) -> bool:
    saved = st.session_state.get(f"{key_prefix}schedule_saved_signature")
    if saved is None:
        return True
    return _schedule_timing_signature(key_prefix) != saved


def _on_schedule_timing_field_change(key_prefix: str) -> None:
    _clamp_frequent_days_state(key_prefix)
    st.session_state.pop(f"{key_prefix}schedule_save_flash", None)


def _clamp_frequent_days_state(key_prefix: str) -> None:
    """Keep the 'more frequent' window strictly lower than the general reminder start."""
    days_key = f"{key_prefix}sched_days"
    frequent_key = f"{key_prefix}sched_frequent_days"
    days_value = int(st.session_state.get(days_key, 30))
    frequent_value = int(st.session_state.get(frequent_key, 7))
    if frequent_value >= days_value:
        st.session_state[frequent_key] = max(1, days_value - 1)


def render_schedule_timing_inputs(
    key_prefix: str,
    *,
    analysis_id: int | None,
    context: dict[str, Any],
) -> None:
    """Single, unified set of reminder-timing controls shared by every channel."""
    st.markdown(f'<div class="reminder-group-title">{t("reminder_timing_group")}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="schedule-timing-panel-marker"></div>', unsafe_allow_html=True)
        days_col, time_col = st.columns([1, 1], gap="medium")
        with days_col:
            st.number_input(
                t("reminder_days"),
                min_value=1,
                max_value=180,
                step=1,
                key=f"{key_prefix}sched_days",
                on_change=_on_schedule_timing_field_change,
                args=(key_prefix,),
            )
        with time_col:
            st.time_input(
                t("reminder_time"),
                key=f"{key_prefix}sched_time",
                on_change=_on_schedule_timing_field_change,
                args=(key_prefix,),
            )

        urgent_col, freq_col, save_col = st.columns([1, 1, 0.55], gap="medium", vertical_alignment="bottom")
        with urgent_col:
            st.markdown('<div class="schedule-save-flash-anchor"></div>', unsafe_allow_html=True)
            st.number_input(
                t("reminder_frequent_days"),
                min_value=1,
                max_value=179,
                step=1,
                key=f"{key_prefix}sched_frequent_days",
                help=t("reminder_frequent_days_help"),
                on_change=_on_schedule_timing_field_change,
                args=(key_prefix,),
            )
            flash = st.session_state.get(f"{key_prefix}schedule_save_flash")
            if flash:
                kind, message = flash
                st.markdown(
                    f'<div class="schedule-save-flash schedule-save-flash-{kind}">{message}</div>',
                    unsafe_allow_html=True,
                )
        with freq_col:
            st.number_input(
                t("reminder_daily_frequency"),
                min_value=1,
                max_value=5,
                step=1,
                key=f"{key_prefix}sched_daily_freq",
                on_change=_on_schedule_timing_field_change,
                args=(key_prefix,),
            )
        with save_col:
            st.markdown('<div class="save-btn-blue-marker"></div>', unsafe_allow_html=True)
            if st.button(
                t("save_btn"),
                key=f"{key_prefix}save_schedule",
                width="stretch",
                disabled=not _schedule_timing_is_dirty(key_prefix),
            ):
                _clamp_frequent_days_state(key_prefix)
                if analysis_id and context is not None:
                    if save_policy_reminder_settings(analysis_id, key_prefix, context):
                        st.session_state[f"{key_prefix}schedule_saved_signature"] = _schedule_timing_signature(key_prefix)
                        st.session_state[f"{key_prefix}schedule_save_flash"] = ("success", t("reminders_saved"))
                        st.rerun()
                else:
                    st.session_state[f"{key_prefix}schedule_save_flash"] = ("warning", t("save_policy_required"))
                    st.rerun()


def format_display_date(value: date) -> str:
    return f"{value.strftime('%B')} {value.day}, {value.year}"


def parse_policy_date(value: str | None) -> date | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"null", "none", "not found", "not_found", "n/a"}:
        return None
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw)
    if iso_match:
        try:
            return datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass
    for fmt in (
        "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
    ):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    dmy = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b", raw)
    if dmy:
        left, middle, year_raw = int(dmy.group(1)), int(dmy.group(2)), int(dmy.group(3))
        year = year_raw + 2000 if year_raw < 100 else year_raw
        for day, month in ((left, middle), (middle, left)):
            try:
                return date(year, month, day)
            except ValueError:
                continue
    return None


def format_policy_date(value: str | None) -> str:
    parsed = parse_policy_date(value)
    return format_display_date(parsed) if parsed else t("not_found")


def analysis_date_present(value: Any) -> bool:
    return parse_policy_date(str(value) if value is not None else None) is not None


def normalize_analysis_date_fields(analysis: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(analysis)
    for key in ("policy_start_date", "policy_end_date", "renewal_date"):
        parsed = parse_policy_date(normalized.get(key))
        if parsed:
            normalized[key] = parsed.isoformat()
        elif normalized.get(key) in (None, "", "null"):
            normalized[key] = None
    return normalized


def _normalize_extracted_date(raw: str | None) -> str | None:
    parsed = parse_policy_date(raw)
    return parsed.isoformat() if parsed else None


def extract_policy_dates_from_text(text: str) -> dict[str, str | None]:
    if not text:
        return {"policy_start_date": None, "policy_end_date": None}
    start_patterns = (
        r"(?:inicio\s+de\s+vigencia|fecha\s+de\s+inicio|vigente\s+desde|vigencia\s+desde|"
        r"desde|inicio|start(?:\s+date)?|effective\s+date|valid\s+from|term\s+from|"
        r"policy\s+period\s+from|beginning|commencement|coverage\s+from)"
        r"[\s:.\-–—]*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})",
    )
    end_patterns = (
        r"(?:fin\s+de\s+vigencia|fecha\s+de\s+(?:fin|término|termino)|vigente\s+hasta|vigencia\s+hasta|"
        r"hasta|fin|término|termino|end(?:\s+date)?|expiration(?:\s+date)?|valid\s+(?:to|until)|"
        r"term\s+to|policy\s+period\s+to|coverage\s+(?:to|until|through))"
        r"[\s:.\-–—]*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})",
    )
    start_date = end_date = None
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_date = _normalize_extracted_date(match.group(1))
            if start_date:
                break
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            end_date = _normalize_extracted_date(match.group(1))
            if end_date:
                break
    return {"policy_start_date": start_date, "policy_end_date": end_date}


def enrich_policy_term_dates(
    analysis: dict[str, Any],
    current_text: str,
    language: str,
    model: Any,
) -> dict[str, Any]:
    enriched = dict(analysis)
    if analysis_date_present(enriched.get("policy_start_date")) and analysis_date_present(enriched.get("policy_end_date")):
        return enriched

    local_dates = extract_policy_dates_from_text(current_text)
    if not analysis_date_present(enriched.get("policy_start_date")) and local_dates.get("policy_start_date"):
        enriched["policy_start_date"] = local_dates["policy_start_date"]
    if not analysis_date_present(enriched.get("policy_end_date")) and local_dates.get("policy_end_date"):
        enriched["policy_end_date"] = local_dates["policy_end_date"]

    if analysis_date_present(enriched.get("policy_start_date")) and analysis_date_present(enriched.get("policy_end_date")):
        return enriched

    try:
        response = model.generate_content(build_policy_term_dates_prompt(current_text, language))
        if not response.candidates:
            return enriched
        raw = response.text
        if not raw or not raw.strip():
            return enriched
        dates = parse_gemini_json(raw.strip())
        for key in ("policy_start_date", "policy_end_date"):
            if not analysis_date_present(enriched.get(key)) and analysis_date_present(dates.get(key)):
                enriched[key] = dates[key]
    except Exception as exc:
        print(f"[enrich_policy_term_dates] {type(exc).__name__}: {exc}")
    return enriched


def get_renewal_alert_days() -> int:
    return int(st.session_state.get("renewal_alert_days", 30))


def get_renewal_red_flag_days() -> int:
    return int(st.session_state.get("renewal_red_flag_days", 15))


def policy_term_dates_available(analysis: dict[str, Any]) -> bool:
    return (
        analysis_date_present(analysis.get("policy_start_date"))
        and analysis_date_present(analysis.get("policy_end_date"))
    )


def format_renewal_date_display(analysis: dict[str, Any]) -> tuple[str, bool]:
    if not policy_term_dates_available(analysis):
        return t("renewal_no_dates"), True
    return format_policy_date(analysis.get("renewal_date")), False


def render_header_date_display() -> None:
    real_today = date.today()
    st.markdown(
        f'<div class="header-date-display">{format_display_date(real_today)}</div>',
        unsafe_allow_html=True,
    )


def render_header_today_and_simulation(config: dict[str, str]) -> None:
    """Today's date with the simulation date + Simulate button right below it (top-right)."""
    real_today = date.today()
    if "reference_date" not in st.session_state:
        st.session_state.reference_date = real_today

    render_header_date_display()

    st.markdown('<div class="header-sim-row-marker"></div>', unsafe_allow_html=True)
    date_col, btn_col = st.columns([1.2, 1], gap="small")
    with date_col:
        st.date_input(
            t("sim_date_label"),
            value=st.session_state.get("reference_date", real_today),
            min_value=real_today,
            key="simulation_date_picker",
            label_visibility="collapsed",
            help=t("sim_date_help"),
        )
    with btn_col:
        if st.button(t("sim_date_btn"), type="primary", width="stretch", key="simulate_btn"):
            picked = st.session_state.simulation_date_picker
            if picked < real_today:
                picked = real_today
            st.session_state.reference_date = picked
            st.session_state.simulation_results = run_simulation_for_all_records(picked, config)
            st.rerun()

    ref = get_reference_date()
    if ref != real_today:
        st.markdown(
            f'<div class="sim-active-banner sim-active-banner-right">{t("sim_date_active").format(date=format_display_date(ref))}</div>',
            unsafe_allow_html=True,
        )


def render_simulation_results_section() -> None:
    results = st.session_state.get("simulation_results")
    if results is not None:
        render_simulation_results(results, get_reference_date())


def render_sidebar_global_contacts() -> None:
    st.markdown(f"**{t('config_global_contacts_title')}**")
    st.text_input(t("email_primary"), key="global_email_primary")
    st.text_input(t("whatsapp_primary"), key="global_whatsapp_primary")
    st.markdown('<div class="sidebar-save-blue-marker save-btn-blue-marker"></div>', unsafe_allow_html=True)
    if st.button(t("save_btn"), key="save_global_contacts_btn", width="stretch"):
        save_global_contacts(
            st.session_state.get("global_email_primary", ""),
            st.session_state.get("global_whatsapp_primary", ""),
        )
        clear_all_reminder_sync_tokens()
        st.success(t("global_contacts_saved"))


def render_sidebar_renewal_alerts() -> None:
    st.markdown(f"**{t('config_renewal_alerts_title')}**")
    st.number_input(
        t("renewal_alert_days_label"),
        min_value=1,
        max_value=365,
        key="renewal_alert_days",
    )
    st.number_input(
        t("renewal_red_flag_days_label"),
        min_value=1,
        max_value=365,
        key="renewal_red_flag_days",
    )


def render_simulation_results(results: list[dict[str, Any]], sim_date: date) -> None:
    sent_total = sum(r.get("sent_count", 0) for r in results)
    st.markdown(
        f'<div class="sim-results-panel">'
        f'<div class="sim-results-title">{t("sim_dispatch_title")} — {format_display_date(sim_date)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    if not results:
        st.info(t("sim_no_records"))
        return
    for item in results:
        css = "sim-results-item" if item.get("sent_count") else "sim-results-muted"
        st.markdown(f'<div class="{css}">{item["summary"]}</div>', unsafe_allow_html=True)
    st.caption(
        t("sim_dispatch_done").format(
            records=len(results),
            date=format_display_date(sim_date),
            count=sent_total,
        )
    )


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #0F172A;
    }
    .stApp { background-color: #EEF2F7 !important; color: #0F172A !important; }
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        background-color: #EEF2F7 !important;
        color: #0F172A !important;
    }
    [data-testid="stMain"] label,
    [data-testid="stMain"] p,
    [data-testid="stMain"] span,
    [data-testid="stMain"] li,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
        color: #0F172A !important;
    }
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p,
    [data-testid="stMain"] small {
        color: #475569 !important;
    }
    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea,
    [data-testid="stMain"] select {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A5F 0%, #152A45 100%) !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .sidebar-title { font-size: 1.45rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.25rem; }
    [data-testid="stSidebar"] .sidebar-tagline { font-size: 0.95rem; color: #CBD5E1; margin-bottom: 0.5rem; }
    [data-testid="stSidebar"] .sidebar-version { font-size: 0.82rem; color: #34D399; margin-bottom: 1rem; font-weight: 600; }
    [data-testid="stSidebar"] .sidebar-step { font-size: 0.9rem; color: #E2E8F0; margin-bottom: 0.55rem; padding: 0.35rem 0 0.35rem 0.65rem; border-left: 3px solid #10B981; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] small { color: #94A3B8 !important; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary { background-color: #243B5C !important; color: #F8FAFC !important; }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] { background-color: #1A3050 !important; color: #E2E8F0 !important; }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] p,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] span { color: #CBD5E1 !important; }
    .header-date-display {
        color: #1E3A5F;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: right;
        line-height: 1.2;
        margin-bottom: 0.55rem;
    }
    .header-sim-row-marker { display: none; }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > .header-sim-row-marker) [data-testid="stHorizontalBlock"] {
        justify-content: flex-end;
    }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > .header-sim-row-marker) [data-testid="stDateInput"] {
        max-width: 9.5rem;
        margin-left: auto;
    }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > .header-sim-row-marker) [data-testid="stDateInput"] input {
        min-height: 2.4rem !important;
        font-weight: 600 !important;
        color: #1E3A5F !important;
        padding-left: 0.65rem !important;
    }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > .header-sim-row-marker) button[data-testid="stBaseButton-primary"] {
        min-height: 2.4rem !important;
    }
    .sim-active-banner-right {
        text-align: right;
        margin-top: 0.5rem;
    }
    .detail-card-value-error {
        color: #991B1B;
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.4;
        word-break: break-word;
    }
    .history-policy-block { margin-bottom: 0.85rem; }
    .upload-compact-marker { display: none; }
    div[data-testid="stColumn"]:has(.upload-compact-marker) [data-testid="stFileUploader"] {
        max-width: 100%;
    }
    div[data-testid="stColumn"]:has(.upload-compact-marker) [data-testid="stFileUploaderDropzone"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        min-height: 0 !important;
        padding: 0.4rem 0.5cm !important;
        background: #BF5700 !important;
        border: 2px solid #A84D00 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stColumn"]:has(.upload-compact-marker) [data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stColumn"]:has(.upload-compact-marker) [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    div[data-testid="stColumn"]:has(.upload-compact-marker) [data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stColumn"]:has(.upload-compact-marker) [data-testid="stFileUploaderDropzone"] button * {
        background: #FFFFFF !important;
        color: #1E3A5F !important;
        border-color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    .upload-file-hint {
        margin-top: 0.35rem;
        margin-bottom: 0.65rem;
    }
    div[data-testid="stColumn"]:has(.upload-compact-marker) [data-testid="stFileUploaderFileData"] {
        width: 100% !important;
        max-width: 100%;
    }
    /* Single-file uploaders: Streamlit still renders a "+" (Add files) after selection */
    [data-testid="stFileUploader"] button[aria-label="Add files"] {
        display: none !important;
    }
    .sim-field-label {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
        line-height: 1.2;
        min-height: 1rem;
    }
    [data-testid="stMain"] button[data-testid="stBaseButton-primary"] p,
    [data-testid="stMain"] button[kind="primary"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    .sim-active-banner {
        color: #1E3A5F;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.85rem;
        padding-top: 0.85rem;
        border-top: 1px solid #E2E8F0;
    }
    .sim-results-panel {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin-bottom: 1.25rem;
    }
    .sim-results-title { color: #166534; font-weight: 700; font-size: 1.05rem; margin-bottom: 0.5rem; }
    .sim-results-item { color: #14532D; font-size: 0.95rem; line-height: 1.55; margin: 0.2rem 0; }
    .sim-results-muted { color: #64748B; font-size: 0.92rem; line-height: 1.5; margin: 0.2rem 0; }
    .panel-title { color: #1E3A5F; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
    .analysis-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
        border: 1px solid #E2E8F0;
    }
    .card-header { color: #1E3A5F; font-size: 1.12rem; font-weight: 700; margin-bottom: 0.75rem; border-bottom: 2px solid #E2E8F0; padding-bottom: 0.5rem; }
    .card-body { color: #334155; font-size: 1rem; line-height: 1.65; }
    .status-green { background: #ECFDF5; border-left: 4px solid #10B981; padding: 0.85rem 1rem; border-radius: 10px; color: #065F46; margin-bottom: 1rem; font-size: 1rem; }
    .status-amber { background: #FFFBEB; border-left: 4px solid #F59E0B; padding: 0.85rem 1rem; border-radius: 10px; color: #92400E; margin-bottom: 1rem; font-size: 1rem; }
    .status-red { background: #FEF2F2; border-left: 4px solid #EF4444; padding: 0.85rem 1rem; border-radius: 10px; color: #991B1B; margin-bottom: 1rem; font-size: 1rem; }
    .policy-holder-line { color: #475569; font-size: 0.98rem; margin: 0.1rem 0 1rem 0; }
    .policy-holder-line strong { color: #1E3A5F; }
    .detail-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        height: 100%;
    }
    .detail-card-label {
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.4rem;
    }
    .detail-card-value {
        color: #1E3A5F;
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.4;
        word-break: break-word;
    }
    .risk-alert-box {
        border-radius: 14px;
        padding: 1.5rem 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        height: calc(100% - 1rem);
        min-height: 5.5rem;
        box-sizing: border-box;
    }
    .risk-alert-text { font-size: 1.5rem; font-weight: 800; letter-spacing: 0.01em; line-height: 1.25; }
    .risk-alert-green { background: #ECFDF5; border: 2px solid #10B981; }
    .risk-alert-green .risk-alert-text { color: #065F46; }
    .risk-alert-yellow { background: #FFFBEB; border: 2px solid #F59E0B; }
    .risk-alert-yellow .risk-alert-text { color: #92400E; }
    .risk-alert-red { background: #FEF2F2; border: 2px solid #EF4444; }
    .risk-alert-red .risk-alert-text { color: #991B1B; }
    .product-title {
        color: #1E3A5F;
        font-size: 2.55rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .main-header { color: #1E3A5F; font-size: 1.55rem; font-weight: 700; margin-bottom: 0.35rem; letter-spacing: -0.01em; }
    .main-subheader { color: #475569; font-size: 1.05rem; margin-bottom: 1.25rem; }
    .question-item { color: #334155; padding: 0.45rem 0; border-bottom: 1px solid #F1F5F9; font-size: 0.98rem; }
    .section-heading { color: #1E3A5F; font-size: 1.25rem; font-weight: 700; margin: 1.25rem 0 0.75rem 0; }
    .version-badge {
        display: inline-block;
        font-size: 2.75rem;
        font-weight: 800;
        color: #1E3A5F;
        line-height: 1;
        margin: 0.25rem 0 0.75rem 0;
        letter-spacing: -0.03em;
    }
    .version-badge-sub {
        color: #64748B;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .duplicate-panel {
        background: #FFF7ED;
        border: 2px solid #F59E0B;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin: 1rem 0 1.25rem 0;
    }
    .duplicate-panel-title { color: #92400E; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
    .duplicate-panel-body { color: #78350F; font-size: 1rem; line-height: 1.55; }
    .save-btn-blue-marker { display: none; }
    div[data-testid="stElementContainer"]:has(.save-btn-blue-marker) + div[data-testid="stElementContainer"] button,
    div[data-testid="stElementContainer"]:has(.save-btn-blue-marker) button {
        background: #FFFFFF !important;
        border: 2px solid #1E3A5F !important;
        color: #1E3A5F !important;
    }
    div[data-testid="stElementContainer"]:has(.save-btn-blue-marker) + div[data-testid="stElementContainer"] button p,
    div[data-testid="stElementContainer"]:has(.save-btn-blue-marker) + div[data-testid="stElementContainer"] button span,
    div[data-testid="stElementContainer"]:has(.save-btn-blue-marker) button p,
    div[data-testid="stElementContainer"]:has(.save-btn-blue-marker) button span,
    div[data-testid="stElementContainer"]:has(.save-btn-blue-marker) button div {
        color: #1E3A5F !important;
        font-weight: 700 !important;
    }
    .whatsapp-demo-link-marker { display: none; }
    .send-now-actions-marker { display: none; }
    div[data-testid="stColumn"]:has(.send-now-actions-marker) [data-testid="stElementContainer"] {
        width: 100% !important;
    }
    div[data-testid="stColumn"]:has(.send-now-actions-marker) button,
    div[data-testid="stColumn"]:has(.send-now-actions-marker) a[data-testid="stLinkButton"] {
        width: 100% !important;
        min-height: 2.75rem !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stColumn"]:has(.send-now-actions-marker) a[data-testid="stLinkButton"] {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-decoration: none !important;
    }
    div[data-testid="stElementContainer"]:has(.whatsapp-demo-link-marker) a[data-testid="stLinkButton"],
    div[data-testid="stElementContainer"]:has(.whatsapp-demo-link-marker) a[data-testid="stLinkButton"] * {
        background: #BF5700 !important;
        border: 2px solid #A84D00 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-save-blue-marker) button,
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-save-blue-marker) + [data-testid="stElementContainer"] button {
        background: #FFFFFF !important;
        border: 2px solid #2563EB !important;
        color: #1D4ED8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-save-blue-marker) button p,
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-save-blue-marker) button span,
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-save-blue-marker) + [data-testid="stElementContainer"] button p,
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-save-blue-marker) + [data-testid="stElementContainer"] button span {
        color: #1D4ED8 !important;
        font-weight: 700 !important;
    }
    .policy-mgmt-radio-marker { display: none; }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .policy-mgmt-radio-marker) [data-testid="stRadio"] label p,
    [data-testid="stVerticalBlock"]:has(.policy-mgmt-radio-marker) [data-testid="stRadio"] label p {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #1E3A5F !important;
    }
    [data-testid="stVerticalBlock"]:has(.policy-mgmt-radio-marker) [data-testid="stRadio"] [role="radiogroup"] {
        gap: 1.5rem;
        margin-top: 0.25rem;
    }
    .history-policy-item-marker { display: none; }
    [data-testid="stVerticalBlock"]:has(.history-policy-item-marker) {
        gap: 0.15rem !important;
        margin-bottom: 1.35rem !important;
        padding-bottom: 0.15rem !important;
    }
    [data-testid="stVerticalBlock"]:has(.history-policy-item-marker) [data-testid="stExpander"] {
        margin-bottom: 0 !important;
    }
    .history-status-section-marker { display: none; }
    [data-testid="stElementContainer"]:has(.history-status-section-marker) {
        margin-top: -0.55rem !important;
        margin-bottom: 0 !important;
    }
    .history-table-head {
        color: #64748B;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 0 0 0.2rem 0;
    }
    .history-status-chip {
        border-radius: 10px;
        padding: 0.45rem 0.5rem;
        text-align: center;
        font-weight: 800;
        font-size: 0.76rem;
        line-height: 1.25;
        min-height: 2.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid;
        margin-top: 0;
        box-sizing: border-box;
    }
    .history-chip-green { background: #ECFDF5; border-color: #10B981; color: #065F46; }
    .history-chip-yellow { background: #FFFBEB; border-color: #F59E0B; color: #92400E; }
    .history-chip-red { background: #FEF2F2; border-color: #EF4444; color: #991B1B; }
    .history-row-marker { display: none; }
    [data-testid="stVerticalBlock"]:has(.history-row-marker) [data-testid="stExpander"] summary {
        padding: 0.45rem 0.75rem !important;
        min-height: unset !important;
        font-size: 0.92rem !important;
    }
    .payment-schedule-marker { display: none; }
    .payment-marker { display: none; }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > .payment-schedule-marker) {
        gap: 0.4rem !important;
    }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > .payment-schedule-marker) [data-testid="stExpander"] summary {
        padding: 0.45rem 0.75rem !important;
        min-height: unset !important;
    }
    [data-testid="stVerticalBlock"]:has(.payment-schedule-marker) [data-testid="stExpander"] summary {
        background-color: #1E3A5F !important;
        color: #FFFFFF !important;
    }
    [data-testid="stElementContainer"]:has(.payment-marker-green) + [data-testid="stElementContainer"] [data-testid="stExpander"] summary,
    div[data-testid="element-container"]:has(.payment-marker-green) + div[data-testid="element-container"] [data-testid="stExpander"] summary {
        border-left: 5px solid #10B981 !important;
    }
    [data-testid="stElementContainer"]:has(.payment-marker-yellow) + [data-testid="stElementContainer"] [data-testid="stExpander"] summary,
    div[data-testid="element-container"]:has(.payment-marker-yellow) + div[data-testid="element-container"] [data-testid="stExpander"] summary {
        border-left: 5px solid #F59E0B !important;
    }
    [data-testid="stElementContainer"]:has(.payment-marker-red) + [data-testid="stElementContainer"] [data-testid="stExpander"] summary,
    div[data-testid="element-container"]:has(.payment-marker-red) + div[data-testid="element-container"] [data-testid="stExpander"] summary {
        border-left: 5px solid #EF4444 !important;
    }
    [data-testid="stVerticalBlock"]:has(.payment-schedule-marker) [data-testid="stExpander"] summary *,
    [data-testid="stVerticalBlock"]:has(.payment-schedule-marker) [data-testid="stExpander"] summary p,
    [data-testid="stVerticalBlock"]:has(.payment-schedule-marker) [data-testid="stExpander"] summary span,
    [data-testid="stVerticalBlock"]:has(.payment-schedule-marker) [data-testid="stExpander"] summary svg,
    [data-testid="stElementContainer"]:has(.payment-marker-green) + [data-testid="stElementContainer"] [data-testid="stExpander"] summary *,
    [data-testid="stElementContainer"]:has(.payment-marker-yellow) + [data-testid="stElementContainer"] [data-testid="stExpander"] summary *,
    [data-testid="stElementContainer"]:has(.payment-marker-red) + [data-testid="stElementContainer"] [data-testid="stExpander"] summary *,
    div[data-testid="element-container"]:has(.payment-marker) + div[data-testid="element-container"] [data-testid="stExpander"] summary * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }
    .payment-body { color: #334155; font-size: 0.95rem; line-height: 1.65; }
    .payment-body strong { color: #1E3A5F; }
    .payment-paid-badge {
        display: inline-block;
        background: #10B981;
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .payment-match-row { padding: 0.3rem 0; font-size: 0.93rem; }
    .payment-match-ok { color: #166534; }
    .payment-match-fail { color: #991B1B; font-weight: 600; }
    .payment-extracted-summary {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0 0.75rem 0;
        color: #1E3A5F;
        font-size: 0.95rem;
        line-height: 1.65;
    }
    .reminder-group-title {
        color: #1E3A5F;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 0.75rem 0;
    }
    .reminder-field-tag {
        color: #64748B;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 0.35rem;
    }
    .reminder-recipient-label {
        font-weight: 600;
        color: #1E3A5F;
        font-size: 0.92rem;
        margin: 0.9rem 0 0.4rem 0;
    }
    .reminder-recipients-compact-marker { display: none; }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.reminder-recipients-compact-marker) {
        padding: 0.55rem 0.75rem 0.65rem !important;
        margin-bottom: 0.35rem !important;
    }
    .schedule-timing-panel-marker { display: none; }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.schedule-timing-panel-marker) {
        overflow: visible !important;
    }
    .schedule-save-flash-anchor { display: none; }
    div[data-testid="stColumn"]:has(.schedule-save-flash-anchor) {
        position: relative !important;
    }
    .schedule-save-flash {
        position: absolute;
        top: calc(100% + 0.2rem);
        left: 0;
        z-index: 20;
        width: max-content;
        max-width: min(36rem, 92vw);
        padding: 0.55rem 0.85rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        line-height: 1.4;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
        pointer-events: none;
    }
    .schedule-save-flash-success {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #6EE7B7;
    }
    .schedule-save-flash-warning {
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FCD34D;
    }
    .schedule-table-marker { display: none; }
    [data-testid="stElementContainer"]:has(.schedule-table-marker) + [data-testid="stElementContainer"] [data-testid="stDataFrame"],
    [data-testid="stElementContainer"]:has(.schedule-table-marker) + [data-testid="stElementContainer"] [data-testid="stDataFrame"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stElementContainer"]:has(.schedule-table-marker) + [data-testid="stElementContainer"] [data-testid="stDataFrame"] {
        scrollbar-width: auto !important;
        scrollbar-color: #94A3B8 #EEF2F7 !important;
    }
    [data-testid="stElementContainer"]:has(.schedule-table-marker) + [data-testid="stElementContainer"] [data-testid="stDataFrame"] *::-webkit-scrollbar {
        width: 12px !important;
        height: 12px !important;
    }
    [data-testid="stElementContainer"]:has(.schedule-table-marker) + [data-testid="stElementContainer"] [data-testid="stDataFrame"] *::-webkit-scrollbar-thumb {
        background-color: #94A3B8 !important;
        border-radius: 6px !important;
    }
    [data-testid="stElementContainer"]:has(.schedule-table-marker) + [data-testid="stElementContainer"] [data-testid="stDataFrame"] *::-webkit-scrollbar-track {
        background-color: #EEF2F7 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.reminder-recipients-compact-marker) [data-testid="stCaptionContainer"] {
        margin-top: 0.2rem;
        margin-bottom: 0;
    }
    .channel-active-banner {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        font-size: 0.92rem;
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        margin: 0.75rem 0 1rem 0;
    }
    .channel-active-email {
        background: #EFF6FF;
        color: #1E3A5F;
        border: 1px solid #BFDBFE;
    }
    .channel-active-whatsapp {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .reminder-config-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.reminder-card-marker) {
        border-radius: 12px !important;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.6rem;
        border-bottom: none;
        background: #F1F5F9;
        border-radius: 12px;
        padding: 0.35rem;
        margin-bottom: 0.25rem;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent;
        color: #475569;
        border-radius: 9px;
        padding: 0.65rem 1.5rem;
        font-weight: 700;
        font-size: 1.02rem;
        border: 2px solid transparent;
        transition: all 0.15s ease;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
        background: #E2E8F0;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button:nth-of-type(1)[aria-selected="true"] {
        background: #1E3A5F !important;
        color: #FFFFFF !important;
        border-color: #1E3A5F !important;
        box-shadow: 0 2px 6px rgba(30, 58, 95, 0.35);
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button:nth-of-type(2)[aria-selected="true"] {
        background: #25D366 !important;
        color: #FFFFFF !important;
        border-color: #25D366 !important;
        box-shadow: 0 2px 6px rgba(37, 211, 102, 0.35);
    }
    .stMainBlockContainer [data-testid="stExpander"] details {
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 0.65rem;
        background-color: #FFFFFF;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }
    .stMainBlockContainer [data-testid="stExpander"] summary {
        background-color: #1E3A5F !important;
        color: #FFFFFF !important;
        padding: 0.85rem 1.1rem;
        font-weight: 600;
        font-size: 1rem;
    }
    .stMainBlockContainer [data-testid="stExpander"] summary *,
    .stMainBlockContainer [data-testid="stExpander"] summary p,
    .stMainBlockContainer [data-testid="stExpander"] summary span,
    .stMainBlockContainer [data-testid="stExpander"] summary div,
    .stMainBlockContainer [data-testid="stExpander"] summary label {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }
    .stMainBlockContainer [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        padding: 1.1rem;
        border-top: 1px solid #E2E8F0;
    }
    .stMainBlockContainer [data-testid="stExpanderDetails"] p,
    .stMainBlockContainer [data-testid="stExpanderDetails"] li,
    .stMainBlockContainer [data-testid="stExpanderDetails"] span,
    .stMainBlockContainer [data-testid="stExpanderDetails"] label,
    .stMainBlockContainer [data-testid="stExpanderDetails"] small,
    .stMainBlockContainer [data-testid="stExpanderDetails"] code {
        color: #0F172A !important;
    }
    [data-testid="stAlert"] p, [data-testid="stAlert"] div { color: inherit !important; }
    button[kind="primary"] {
        background-color: #1E3A5F !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"] {
        color: #1E3A5F !important;
        border: 1px solid #CBD5E1 !important;
        background: #FFFFFF !important;
    }
</style>
"""


def render_version_badge(version: int, insurance_id: str = "") -> None:
    st.markdown(f'<div class="version-badge">#{version}</div>', unsafe_allow_html=True)
    sub = t("version_label")
    if insurance_id:
        sub = f"{sub} · {insurance_id}"
    st.markdown(f'<div class="version-badge-sub">{sub}</div>', unsafe_allow_html=True)


def persist_analysis_to_db(analysis: dict[str, Any], context: dict[str, Any], upload_version: int) -> int:
    insurance_id = get_insurance_id(analysis)
    freq = normalize_payment_frequency(analysis.get("payment_frequency"))
    email_settings = {
        "payment_frequency": freq,
        "reminder_start_days": int(analysis.get("suggested_reminder_start_days", 30)),
        "reminder_base_time": analysis.get("suggested_reminder_time", "09:00"),
        "primary": st.session_state.get("current_email_primary", ""),
        "secondary": st.session_state.get("current_email_secondary", ""),
    }
    whatsapp_settings = {
        "payment_frequency": freq,
        "reminder_start_days": email_settings["reminder_start_days"],
        "reminder_base_time": email_settings["reminder_base_time"],
        "primary": "",
        "secondary": "",
    }
    reminder_blob = build_reminder_settings_blob(email_settings, whatsapp_settings)
    row_id = save_to_database(
        policy_holder=analysis.get("policy_holder", ""),
        insurer=analysis.get("insurer", ""),
        renewal_date=analysis.get("renewal_date") or "",
        premium=analysis.get("premium_amount", ""),
        email_primary=email_settings["primary"],
        email_secondary=email_settings["secondary"],
        language=context.get("analysis_language", st.session_state.language),
        analysis=analysis,
        context=context,
        payment_frequency=freq,
        reminder_start_days=email_settings["reminder_start_days"],
        reminder_base_time=email_settings["reminder_base_time"],
        insurance_id=insurance_id,
        upload_version=upload_version,
        reminder_settings_json=json.dumps(reminder_blob),
    )
    st.session_state.saved_analysis_id = row_id
    context["analysis_id"] = row_id
    context["insurance_id"] = insurance_id
    context["upload_version"] = upload_version
    st.session_state.analysis_context = context
    st.session_state.analysis_db_saved = True
    return row_id


def render_duplicate_alert_panel() -> None:
    if not st.session_state.get("duplicate_pending"):
        return
    analysis = st.session_state.last_analysis
    context = st.session_state.get("analysis_context", {})
    insurance_id = get_insurance_id(analysis)
    existing_count = int(st.session_state.get("duplicate_existing_count") or 0)
    pending_version = int(context.get("pending_upload_version") or 1)
    policy_display = (analysis.get("policy_number") or insurance_id or t("not_found")).strip()

    st.markdown(
        f'<div class="duplicate-panel">'
        f'<div class="duplicate-panel-title">{t("duplicate_title")}</div>'
        f'<div class="duplicate-panel-body">'
        f'{t("duplicate_id_label")}: <strong>{policy_display}</strong><br>'
        f'{t("duplicate_message").format(count=existing_count)}'
        f"</div></div>",
        unsafe_allow_html=True,
    )
    render_version_badge(pending_version, insurance_id)
    col_keep, col_replace = st.columns(2)
    with col_keep:
        if st.button(t("duplicate_keep_both"), type="primary", key="dup_keep_both"):
            persist_analysis_to_db(analysis, context, pending_version)
            st.session_state.duplicate_pending = False
            st.session_state.pop("duplicate_existing_count", None)
            st.success(t("duplicate_saved_both").format(version=pending_version))
            st.rerun()
    with col_replace:
        if st.button(t("duplicate_want_replace"), key="dup_want_replace"):
            clear_current_analysis()
            st.session_state.replace_manual_flash = True
            st.rerun()
    st.info(t("duplicate_not_saved"))
    st.markdown("---")


def render_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="analysis-card"><div class="card-header">{title}</div>'
        f'<div class="card-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def render_status_indicator(risk_level: str) -> None:
    labels = {"critical": ("status-red", "status_overdue"), "warning": ("status-amber", "status_pending")}
    css, key = labels.get(risk_level, ("status-green", "status_confirmed"))
    st.markdown(f'<div class="{css}"><strong>{t(key)}</strong></div>', unsafe_allow_html=True)


def render_detail_card(label: str, value: str, *, error: bool = False) -> None:
    value_class = "detail-card-value-error" if error else "detail-card-value"
    safe_value = xml.sax.saxutils.escape(str(value))
    st.markdown(
        f'<div class="detail-card"><div class="detail-card-label">{label}</div>'
        f'<div class="{value_class}">{safe_value}</div></div>',
        unsafe_allow_html=True,
    )


def render_detail_alert_box(color: str, text: str) -> None:
    css_class = {
        "green": "risk-alert-green",
        "yellow": "risk-alert-yellow",
        "red": "risk-alert-red",
    }.get(color, "risk-alert-green")
    safe_text = xml.sax.saxutils.escape(text)
    st.markdown(
        f'<div class="risk-alert-box {css_class}">'
        f'<div class="risk-alert-text">{safe_text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_risk_alert_box(risk_level: str) -> None:
    mapping = {
        "ok": ("risk-alert-green", "status_on_time"),
        "warning": ("risk-alert-yellow", "status_warning_approaching"),
        "critical": ("risk-alert-red", "status_do_it_now"),
    }
    css_class, text_key = mapping.get(risk_level, mapping["ok"])
    st.markdown(
        f'<div class="risk-alert-box {css_class}">'
        f'<div class="risk-alert-text">{t(text_key)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def extract_first_amount(text: str) -> float | None:
    match = re.search(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+", text or "")
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def extract_currency_label(text: str) -> str:
    match = re.search(r"(MXN|USD|EUR|\$|€)", text or "", re.IGNORECASE)
    return match.group() if match else ""


def format_currency_amount(value: float, currency: str) -> str:
    formatted = f"{value:,.2f}"
    if currency == "$":
        return f"${formatted}"
    if currency:
        return f"{formatted} {currency.upper()}"
    return formatted


def compute_premium_per_term(premium_amount: str, freq: str) -> str:
    total = extract_first_amount(premium_amount)
    if total is None:
        return t("not_found")
    count = payments_per_year(freq) or 1
    currency = extract_currency_label(premium_amount)
    return format_currency_amount(total / count, currency)


def get_policy_renewal_due_value(analysis: dict[str, Any]) -> str | None:
    parsed = parse_policy_date(analysis.get("policy_end_date"))
    return parsed.isoformat() if parsed else None


def format_next_policy_renewal_date_display(analysis: dict[str, Any]) -> tuple[str, bool]:
    if not policy_term_dates_available(analysis):
        return t("renewal_no_dates"), True
    return format_policy_date(analysis.get("policy_end_date")), False


def payment_alert_text(color: str) -> str:
    return {
        "green": t("payment_alert_on_time"),
        "yellow": t("payment_alert_pay"),
        "red": t("payment_alert_pay_now"),
    }.get(color, t("payment_alert_on_time"))


def get_payment_alert_status(
    analysis: dict[str, Any],
    context: dict[str, Any],
    reference: date | None = None,
) -> tuple[str, str]:
    ref = reference or get_reference_date()
    due = get_next_unpaid_payment_date(analysis, context, ref)
    if not due:
        renewal = analysis.get("renewal_date") or ""
        if not renewal or renewal == "null":
            return "yellow", t("payment_alert_pay")
        try:
            due = datetime.strptime(renewal[:10], "%Y-%m-%d").date()
        except ValueError:
            return "yellow", t("payment_alert_pay")

    confirmations = context.get("payment_confirmations") or {}
    if is_term_paid_record(confirmations.get(due.isoformat())):
        return "green", t("payment_alert_on_time")

    color = payment_due_color(due, ref)
    return color, payment_alert_text(color)


def render_policy_detail_grid(
    analysis: dict[str, Any],
    context: dict[str, Any],
) -> None:
    freq = normalize_payment_frequency(
        context.get("payment_frequency") or analysis.get("payment_frequency")
    )

    row1 = st.columns(3, gap="medium")
    with row1[0]:
        render_detail_card(t("field_plan_id"), analysis.get("policy_number") or t("not_found"))
    with row1[1]:
        render_detail_card(t("field_insurance_type"), analysis.get("insurance_type") or t("not_found"))
    with row1[2]:
        render_detail_card(t("field_insurance_company"), analysis.get("insurer") or t("not_found"))

    row2 = st.columns(3, gap="medium")
    with row2[0]:
        render_detail_card(t("field_payment_frequency"), freq_label(freq))
    with row2[1]:
        render_detail_card(t("plan_start_date_label"), format_policy_date(analysis.get("policy_start_date")))
    with row2[2]:
        render_detail_card(t("plan_end_date_label"), format_policy_date(analysis.get("policy_end_date")))

    premium_total = analysis.get("premium_amount") or t("not_found")
    row3 = st.columns(2, gap="medium")
    with row3[0]:
        render_detail_card(t("renewal_premium_total_label"), premium_total)
    with row3[1]:
        render_detail_card(t("premium_per_term_label"), compute_premium_per_term(premium_total, freq))

    row4 = st.columns(2, gap="medium")
    with row4[0]:
        next_due = get_next_unpaid_payment_date(analysis, context)
        payment_due_text = (
            format_display_date(next_due)
            if next_due
            else format_policy_date(analysis.get("renewal_date"))
        )
        render_detail_card(t("next_payment_due_date_label"), payment_due_text)
    with row4[1]:
        payment_color, payment_alert = get_payment_alert_status(analysis, context)
        render_detail_alert_box(payment_color, payment_alert)

    row5 = st.columns(2, gap="medium")
    with row5[0]:
        renewal_date_text, renewal_date_error = format_next_policy_renewal_date_display(analysis)
        render_detail_card(t("next_renewal_date_label"), renewal_date_text, error=renewal_date_error)
    with row5[1]:
        renewal_color, renewal_alert = get_history_renewal_status(analysis, get_reference_date())
        render_detail_alert_box(renewal_color, renewal_alert)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            policy_holder TEXT,
            insurer TEXT,
            renewal_date TEXT,
            premium TEXT,
            analyzed_at TEXT,
            email_primary TEXT,
            email_secondary TEXT,
            language TEXT,
            analysis_json TEXT,
            context_json TEXT,
            payment_frequency TEXT,
            reminder_start_days INTEGER,
            reminder_base_time TEXT,
            reminders_active INTEGER DEFAULT 1,
            payment_confirmed_at TEXT,
            reminder_settings_json TEXT,
            insurance_id TEXT,
            upload_version INTEGER DEFAULT 1,
            payment_confirmations_json TEXT
        )
        """
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(analyses)").fetchall()}
    migrations = {
        "analysis_json": "ALTER TABLE analyses ADD COLUMN analysis_json TEXT",
        "context_json": "ALTER TABLE analyses ADD COLUMN context_json TEXT",
        "payment_frequency": "ALTER TABLE analyses ADD COLUMN payment_frequency TEXT",
        "reminder_start_days": "ALTER TABLE analyses ADD COLUMN reminder_start_days INTEGER",
        "reminder_base_time": "ALTER TABLE analyses ADD COLUMN reminder_base_time TEXT",
        "reminders_active": "ALTER TABLE analyses ADD COLUMN reminders_active INTEGER DEFAULT 1",
        "payment_confirmed_at": "ALTER TABLE analyses ADD COLUMN payment_confirmed_at TEXT",
        "reminder_settings_json": "ALTER TABLE analyses ADD COLUMN reminder_settings_json TEXT",
        "insurance_id": "ALTER TABLE analyses ADD COLUMN insurance_id TEXT",
        "upload_version": "ALTER TABLE analyses ADD COLUMN upload_version INTEGER DEFAULT 1",
        "payment_confirmations_json": "ALTER TABLE analyses ADD COLUMN payment_confirmations_json TEXT",
    }
    for col, sql in migrations.items():
        if col not in existing:
            conn.execute(sql)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()
    conn.close()


GLOBAL_EMAIL_KEY = "global_email_primary"
GLOBAL_WHATSAPP_KEY = "global_whatsapp_primary"


def load_global_contacts() -> dict[str, str]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT key, value FROM app_settings WHERE key IN (?, ?)",
        (GLOBAL_EMAIL_KEY, GLOBAL_WHATSAPP_KEY),
    ).fetchall()
    conn.close()
    data = {key: value for key, value in rows}
    return {
        "email_primary": (data.get(GLOBAL_EMAIL_KEY) or "").strip(),
        "whatsapp_primary": (data.get(GLOBAL_WHATSAPP_KEY) or "").strip(),
    }


def save_global_contacts(email_primary: str, whatsapp_primary: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    for key, value in (
        (GLOBAL_EMAIL_KEY, email_primary.strip()),
        (GLOBAL_WHATSAPP_KEY, whatsapp_primary.strip()),
    ):
        conn.execute(
            """
            INSERT INTO app_settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
    conn.commit()
    conn.close()


def clear_all_reminder_sync_tokens() -> None:
    for key in list(st.session_state.keys()):
        if str(key).endswith("reminder_sync_token"):
            st.session_state.pop(key, None)


def resolve_contact_primary(
    channel: str,
    channel_data: dict[str, Any],
    contacts: dict[str, str],
    global_contacts: dict[str, str],
    settings: dict[str, Any],
) -> str:
    if isinstance(settings.get(channel), dict):
        policy_primary = (channel_data.get("primary") or "").strip()
        if policy_primary:
            return policy_primary
        if channel == "email":
            return (global_contacts.get("email_primary") or "").strip()
        return (global_contacts.get("whatsapp_primary") or "").strip()

    if channel == "email":
        candidates = [
            (global_contacts.get("email_primary") or "").strip(),
            (contacts.get("email_primary") or "").strip(),
            (channel_data.get("primary") or "").strip(),
        ]
    else:
        candidates = [
            (global_contacts.get("whatsapp_primary") or "").strip(),
            (contacts.get("phone") or "").strip(),
            (channel_data.get("primary") or "").strip(),
        ]
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def resolve_contact_secondary(
    channel: str,
    channel_data: dict[str, Any],
    contacts: dict[str, str],
) -> str:
    if channel == "email":
        return (channel_data.get("secondary") or contacts.get("email_secondary") or "").strip()
    return (channel_data.get("secondary") or contacts.get("phone_secondary") or "").strip()


def save_policy_reminder_settings(
    analysis_id: int | None,
    key_prefix: str,
    context: dict[str, Any],
) -> bool:
    if not analysis_id:
        return False
    email_settings = read_channel_form_values(key_prefix, "email", context=context)
    whatsapp_settings = read_channel_form_values(key_prefix, "whatsapp", context=context)
    settings = build_reminder_settings_blob(email_settings, whatsapp_settings)
    context.update({
        "payment_frequency": email_settings["payment_frequency"],
        "reminder_start_days": int(email_settings["reminder_start_days"]),
        "reminder_base_time": email_settings["reminder_base_time"],
        "email_primary": email_settings["primary"],
        "email_secondary": email_settings["secondary"],
        "whatsapp_phone": whatsapp_settings["primary"],
        "whatsapp_secondary": whatsapp_settings["secondary"],
    })
    update_analysis_record(
        int(analysis_id),
        payment_frequency=email_settings["payment_frequency"],
        reminder_start_days=int(email_settings["reminder_start_days"]),
        reminder_base_time=email_settings["reminder_base_time"],
        email_primary=email_settings["primary"],
        email_secondary=email_settings["secondary"],
        reminder_settings_json=json.dumps(settings),
        context_json=json.dumps(context),
    )
    st.session_state.pop(f"{key_prefix}reminder_sync_token", None)
    if st.session_state.get("saved_analysis_id") == analysis_id:
        st.session_state.analysis_context = context
    return True


def normalize_insurance_id(raw: str) -> str:
    cleaned = re.sub(r"[\s\-_/().]", "", raw.strip().upper())
    return cleaned


def get_insurance_id(analysis: dict[str, Any]) -> str:
    """Stable id for duplicate detection — policy number preferred."""
    for key in ("policy_number", "insurance_id", "policy_id"):
        value = (analysis.get(key) or "").strip()
        if value and value.lower() not in ("null", "not found", "n/a", "na"):
            return normalize_insurance_id(value)
    holder = (analysis.get("policy_holder") or "").strip().lower()
    insurer = (analysis.get("insurer") or "").strip().lower()
    if holder and insurer:
        return normalize_insurance_id(f"{insurer}:{holder}")
    return ""


_INVALID_ID_VALUES = frozenset({"null", "not found", "n/a", "na", ""})


def insurance_id_from_row(row: dict[str, Any]) -> str:
    stored = normalize_insurance_id(row.get("insurance_id") or "")
    if stored:
        return stored
    if row.get("analysis_json"):
        try:
            return get_insurance_id(json.loads(row["analysis_json"]))
        except json.JSONDecodeError:
            pass
    return get_insurance_id(
        {"policy_holder": row.get("policy_holder", ""), "insurer": row.get("insurer", "")}
    )


def get_policy_number_display(row: dict[str, Any]) -> str:
    """Human-readable policy number for list labels."""
    if row.get("analysis_json"):
        try:
            analysis = json.loads(row["analysis_json"])
            for key in ("policy_number", "policy_id", "insurance_id"):
                value = (analysis.get(key) or "").strip()
                if value and value.lower() not in _INVALID_ID_VALUES:
                    return value
        except json.JSONDecodeError:
            pass
    stored = (row.get("insurance_id") or "").strip()
    if stored:
        return stored
    fallback = insurance_id_from_row(row)
    return fallback if fallback else t("not_found")


def get_row_insurance_type(row: dict[str, Any]) -> str:
    if row.get("analysis_json"):
        try:
            analysis = json.loads(row["analysis_json"])
            ins_type = (analysis.get("insurance_type") or "").strip()
            if ins_type and ins_type.lower() not in _INVALID_ID_VALUES:
                return ins_type
        except json.JSONDecodeError:
            pass
    return t("not_found")


def format_policy_header_label(
    *,
    policy_num: str,
    holder: str,
    insurance_type: str,
    renewal_date: str | None,
    version: int = 1,
    paid_tag: str = "",
) -> str:
    version_tag = f" #{version}" if version > 1 else ""
    renewal_display = format_policy_date(renewal_date)
    ins_type = (insurance_type or "").strip() or t("not_found")
    return f"{policy_num}{version_tag} — {holder} · {ins_type} · {renewal_display}{paid_tag}"


def format_history_label_compact(row: dict[str, Any]) -> str:
    policy_num = get_policy_number_display(row)
    version = int(row.get("upload_version") or 1)
    holder = row.get("policy_holder") or t("not_found")
    paid_tag = "" if row.get("reminders_active", 1) else " ✓"
    return format_policy_header_label(
        policy_num=policy_num,
        holder=holder,
        insurance_type=get_row_insurance_type(row),
        renewal_date=row.get("renewal_date"),
        version=version,
        paid_tag=paid_tag,
    )


def format_history_label(row: dict[str, Any]) -> str:
    policy_num = get_policy_number_display(row)
    version = int(row.get("upload_version") or 1)
    analyzed = (row.get("analyzed_at") or "")[:10]
    holder = row.get("policy_holder") or t("not_found")
    status = "" if row.get("reminders_active", 1) else " ✓"
    header = format_policy_header_label(
        policy_num=policy_num,
        holder=holder,
        insurance_type=get_row_insurance_type(row),
        renewal_date=row.get("renewal_date"),
        version=version,
    )
    return f"{header} · {analyzed}{status}"


def collect_row_insurance_ids(row: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    stored = normalize_insurance_id(row.get("insurance_id") or "")
    if stored:
        ids.add(stored)
    if row.get("analysis_json"):
        try:
            analysis = json.loads(row["analysis_json"])
            derived = get_insurance_id(analysis)
            if derived:
                ids.add(derived)
            policy_number = (analysis.get("policy_number") or "").strip()
            if policy_number.lower() not in _INVALID_ID_VALUES:
                ids.add(normalize_insurance_id(policy_number))
        except json.JSONDecodeError:
            pass
    fallback = get_insurance_id(
        {"policy_holder": row.get("policy_holder", ""), "insurer": row.get("insurer", "")}
    )
    if fallback:
        ids.add(fallback)
    return ids


def find_existing_uploads(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    target = get_insurance_id(analysis)
    if not target:
        return []
    matches: list[dict[str, Any]] = []
    seen_row_ids: set[int] = set()
    for row in get_history():
        row_id = int(row["id"])
        if row_id in seen_row_ids:
            continue
        if target in collect_row_insurance_ids(row):
            matches.append(row)
            seen_row_ids.add(row_id)
    return matches


def next_upload_version(analysis: dict[str, Any]) -> int:
    rows = find_existing_uploads(analysis)
    if not rows:
        return 1
    versions = [int(r.get("upload_version") or 1) for r in rows]
    return max(versions) + 1


def backfill_insurance_ids() -> None:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, analysis_json FROM analyses WHERE insurance_id IS NULL OR insurance_id = ''"
    ).fetchall()
    for row_id, analysis_json in rows:
        if not analysis_json:
            continue
        try:
            insurance_id = get_insurance_id(json.loads(analysis_json))
        except json.JSONDecodeError:
            continue
        if insurance_id:
            conn.execute("UPDATE analyses SET insurance_id = ? WHERE id = ?", (insurance_id, row_id))
    conn.commit()
    conn.close()


def save_to_database(
    policy_holder: str,
    insurer: str,
    renewal_date: str,
    premium: str,
    email_primary: str,
    email_secondary: str,
    language: str,
    analysis: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    payment_frequency: str = "annual",
    reminder_start_days: int = 30,
    reminder_base_time: str = "09:00",
    insurance_id: str = "",
    upload_version: int = 1,
    reminder_settings_json: str = "{}",
) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """
        INSERT INTO analyses
            (policy_holder, insurer, renewal_date, premium, analyzed_at,
             email_primary, email_secondary, language, analysis_json, context_json,
             payment_frequency, reminder_start_days, reminder_base_time,
             reminders_active, reminder_settings_json, insurance_id, upload_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
        """,
        (
            policy_holder, insurer, renewal_date, premium,
            datetime.now().isoformat(), email_primary, email_secondary, language,
            json.dumps(analysis) if analysis else None,
            json.dumps(context) if context else None,
            payment_frequency, reminder_start_days, reminder_base_time,
            reminder_settings_json,
            insurance_id or None,
            upload_version,
        ),
    )
    row_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()
    return row_id


def update_analysis_record(analysis_id: int, **fields: Any) -> None:
    if not fields:
        return
    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [analysis_id]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"UPDATE analyses SET {columns} WHERE id = ?", values)
    conn.commit()
    conn.close()


def delete_from_database(analysis_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()


def get_history() -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM analyses ORDER BY analyzed_at DESC LIMIT 50").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_analysis_by_id(analysis_id: int) -> dict[str, Any] | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def load_payment_confirmations(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row or not row.get("payment_confirmations_json"):
        return {}
    try:
        return json.loads(row["payment_confirmations_json"])
    except json.JSONDecodeError:
        return {}


def save_payment_confirmation(analysis_id: int, due_date_iso: str, record: dict[str, Any]) -> None:
    row = get_analysis_by_id(analysis_id)
    confirmations = load_payment_confirmations(row)
    if record.get("status") == "confirmed":
        record = {**record, "paid": True, "due_date": due_date_iso}
    confirmations[due_date_iso] = record
    update_analysis_record(analysis_id, payment_confirmations_json=json.dumps(confirmations))


# ---------------------------------------------------------------------------
# PDF & AI
# ---------------------------------------------------------------------------
def extract_text(pdf_file) -> tuple[str | None, str | None, str | None]:
    try:
        pages_text: list[str] = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text.strip())
        combined = "\n\n".join(pages_text).strip()
        if not combined:
            return None, "err_empty_pdf", "No extractable text was found in this PDF."
        return combined, None, None
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        return None, "err_pdf_unreadable", detail


REQUIRED_ANALYSIS_FIELDS: dict[str, str] = {
    "policy_number": "field_plan_id",
    "payment_frequency": "field_payment_frequency",
    "premium_amount": "field_total_amount",
    "insurance_type": "field_insurance_type",
    "insurer": "field_insurance_company",
    "policy_start_date": "plan_start_date_label",
    "policy_end_date": "plan_end_date_label",
}


DATE_ANALYSIS_FIELDS = frozenset({"policy_start_date", "policy_end_date"})


def analysis_field_present(value: str | None) -> bool:
    cleaned = (value or "").strip()
    return bool(cleaned) and cleaned.lower() not in _INVALID_ID_VALUES


def normalize_policy_holder(analysis: dict[str, Any]) -> dict[str, Any]:
    if not analysis_field_present(analysis.get("policy_holder")):
        analysis["policy_holder"] = "****"
    return analysis


def get_missing_required_fields(analysis: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field_key in REQUIRED_ANALYSIS_FIELDS:
        if field_key == "payment_frequency":
            raw = (analysis.get("payment_frequency") or "").strip()
            if not analysis_field_present(raw):
                missing.append(field_key)
                continue
            normalized = normalize_payment_frequency(raw)
            if normalized not in PAYMENT_FREQUENCIES:
                missing.append(field_key)
            else:
                analysis["payment_frequency"] = normalized
        elif field_key in DATE_ANALYSIS_FIELDS:
            if not analysis_date_present(analysis.get(field_key)):
                missing.append(field_key)
        elif not analysis_field_present(analysis.get(field_key)):
            missing.append(field_key)
    return missing


def merge_manual_analysis_fields(analysis: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    merged = dict(analysis)
    for key, value in values.items():
        if key in DATE_ANALYSIS_FIELDS:
            if isinstance(value, date):
                merged[key] = value.isoformat()
            elif value is not None and str(value).strip():
                parsed = parse_policy_date(str(value))
                merged[key] = parsed.isoformat() if parsed else str(value).strip()
            continue
        cleaned = str(value).strip() if value is not None else ""
        if cleaned:
            merged[key] = cleaned
    if "payment_frequency" in values and str(values.get("payment_frequency") or "").strip():
        merged["payment_frequency"] = normalize_payment_frequency(str(values["payment_frequency"]))
    return normalize_policy_holder(merged)


def reset_upload_state() -> None:
    st.session_state.upload_widget_key = st.session_state.get("upload_widget_key", 0) + 1
    for key in (
        "upload_parse_error",
        "pending_analysis",
        "pending_missing_fields",
        "last_analysis",
        "analysis_context",
        "saved_analysis_id",
        "analysis_db_saved",
        "duplicate_pending",
        "duplicate_existing_count",
    ):
        st.session_state.pop(key, None)


@st.dialog(" ")
def upload_parse_error_dialog() -> None:
    st.markdown(f"**{t('upload_error_title')}**")
    error = st.session_state.get("upload_parse_error", {})
    st.error(error.get("message", t("err_pdf_unreadable")))
    detail = error.get("detail", "")
    if detail:
        st.markdown(f"**{t('upload_error_detail_label')}:** {detail}")
    if st.button(t("upload_error_ok"), type="primary", width="stretch"):
        st.session_state.pop("upload_parse_error", None)
        reset_upload_state()
        st.rerun()


def render_missing_fields_form() -> None:
    analysis = st.session_state.get("pending_analysis") or {}
    missing_keys: list[str] = st.session_state.get("pending_missing_fields") or []
    if not missing_keys:
        return

    st.warning(t("upload_missing_title"))
    st.caption(t("upload_missing_hint"))
    for field_key in missing_keys:
        label_key = REQUIRED_ANALYSIS_FIELDS[field_key]
        st.markdown(t("upload_missing_field_item").format(field=t(label_key)))

    with st.form("manual_missing_fields"):
        inputs: dict[str, Any] = {}
        for field_key in missing_keys:
            label_key = REQUIRED_ANALYSIS_FIELDS[field_key]
            if field_key in DATE_ANALYSIS_FIELDS:
                existing = parse_policy_date(analysis.get(field_key))
                inputs[field_key] = st.date_input(t(label_key), value=existing or date.today())
            else:
                help_text = t("field_payment_frequency_help") if field_key == "payment_frequency" else None
                inputs[field_key] = st.text_input(t(label_key), help=help_text)
        col_continue, col_retry = st.columns(2)
        with col_continue:
            continue_btn = st.form_submit_button(t("upload_manual_continue"), type="primary")
        with col_retry:
            retry_btn = st.form_submit_button(t("upload_try_again"))

    if retry_btn:
        st.session_state.pop("pending_analysis", None)
        st.session_state.pop("pending_missing_fields", None)
        reset_upload_state()
        st.rerun()

    if continue_btn:
        merged = merge_manual_analysis_fields(analysis, inputs)
        still_missing = get_missing_required_fields(merged)
        if still_missing:
            st.session_state.pending_analysis = merged
            st.session_state.pending_missing_fields = still_missing
            st.error(t("upload_missing_title"))
            st.rerun()
            return
        st.session_state.pop("pending_analysis", None)
        st.session_state.pop("pending_missing_fields", None)
        renewal_date = merged.get("renewal_date") or ""
        risk_level, risk_msg_key = compute_risk_level(
            renewal_date if renewal_date != "null" else None,
            get_reference_date(),
        )
        saved = process_new_analysis(merged, risk_level, risk_msg_key)
        if saved:
            st.success(t("analysis_saved"))
        st.rerun()


def build_analysis_prompt(current_text: str, language: str) -> str:
    lang_instruction = (
        "Respond entirely in English." if language == "en"
        else "Responde completamente en español."
    )
    return f"""
You are an expert insurance analyst for Mexican Gastos Médicos Mayores (GMM) policies.
Focus ONLY on renewal and payment — no year-over-year comparison.

Return ONLY valid JSON:
{{
  "policy_holder": "full name",
  "insurer": "company name",
  "policy_number": "policy or certificate number from the document",
  "insurance_type": "type of insurance product (e.g. Gastos Médicos Mayores, GMM, life, auto)",
  "policy_start_date": "YYYY-MM-DD full policy term start date, or null",
  "policy_end_date": "YYYY-MM-DD full policy term end date, or null",
  "renewal_date": "YYYY-MM-DD next payment due date or null",
  "premium_amount": "total renewal premium amount with currency",
  "payment_frequency": "monthly|quarterly|semester|annual",
  "coverage_details": "key coverage bullet points",
  "smart_questions": ["3-5 questions about renewal and payment"],
  "suggested_reminder_start_days": 30,
  "suggested_reminder_time": "09:00",
  "reminder_rationale": "brief explanation of why this schedule fits the payment type (do not mention premium amounts)"
}}

Rules:
- {lang_instruction}
- CRITICAL — policy term dates: You MUST extract policy_start_date and policy_end_date (full plan/policy term, NOT the next payment date).
  These dates are almost always in the document. Search the entire text carefully before returning null.
  Look for labels and phrases such as:
  * Spanish: "vigencia", "inicio de vigencia", "fin de vigencia", "fecha de inicio", "fecha de fin",
    "vigente desde", "vigente hasta", "desde", "hasta", "periodo de vigencia", "vigencia del plan",
    "término", "termino", "fecha de emisión", "duración de la póliza"
  * English: "effective date", "expiration date", "policy period", "term from", "term to",
    "valid from", "valid until", "coverage period", "start date", "end date", "beginning", "commencement"
  Dates may appear as DD/MM/YYYY, MM/DD/YYYY, or written out. Always normalize to YYYY-MM-DD in JSON.
- Detect payment_frequency from policy (mensual/trimestral/semestral/anual).
- Suggest reminder_start_days based on payment type:
  * annual: 30-45 days before
  * semester: 21-30 days before
  * quarterly: 14-21 days before
  * monthly: 7-10 days before
- In reminder_rationale, explain timing only — do NOT repeat premium amounts; they are shown elsewhere in the app.
- Return ONLY JSON, no markdown fences.

POLICY TEXT:
{current_text[:15000]}
"""


def build_policy_term_dates_prompt(current_text: str, language: str) -> str:
    lang_instruction = (
        "Respond entirely in English." if language == "en"
        else "Responde completamente en español."
    )
    return f"""
You are re-analyzing a Mexican insurance policy PDF because the policy term start/end dates were not found on the first pass.
Your ONLY job is to locate the full policy/plan term start date and end date. They ARE in this document.

Return ONLY valid JSON:
{{
  "policy_start_date": "YYYY-MM-DD or null",
  "policy_end_date": "YYYY-MM-DD or null"
}}

Search strategy:
1. Scan for explicit labels: vigencia, inicio/fin de vigencia, vigente desde/hasta, desde/hasta, effective date,
   expiration date, policy period, term from/to, valid from/until, coverage period, start/end date.
2. If labels differ, look for synonymous wording: beginning, commencement, término, termino, duración,
   periodo del plan, fecha de emisión paired with a later end date, "del ... al ...", "from ... to ...".
3. Prefer the full policy/plan coverage window — NOT the next premium due date and NOT payment receipt dates.
4. Dates may use DD/MM/YYYY, MM/DD/YYYY, or spelled-out month names. Normalize to YYYY-MM-DD.
5. Only return null if the document truly contains no policy term boundaries after an exhaustive search.

Rules:
- {lang_instruction}
- Return ONLY JSON, no markdown fences.

POLICY TEXT:
{current_text[:15000]}
"""


def parse_gemini_json(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def analyze_with_gemini(
    current_text: str, language: str = "en"
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    api_key = get_api_key()
    if not api_key:
        return None, "err_no_api_key", None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(get_gemini_model())
        response = model.generate_content(build_analysis_prompt(current_text, language))
        if not response.candidates:
            return None, "err_api_failure", "Gemini returned no candidates."
        raw = response.text
        if not raw or not raw.strip():
            return None, "err_api_failure", "Empty response from Gemini."
        analysis = parse_gemini_json(raw.strip())
        analysis = enrich_policy_term_dates(analysis, current_text, language, model)
        analysis = normalize_analysis_date_fields(analysis)
        return analysis, None, None
    except Exception as exc:
        debug = f"{type(exc).__name__}: {exc}"
        print(f"[analyze_with_gemini] {debug}")
        traceback.print_exc()
        return None, "err_api_failure", debug


def build_payment_proof_prompt(text: str, language: str) -> str:
    lang_instruction = (
        "Respond entirely in English." if language == "en"
        else "Responde completamente en español."
    )
    return f"""
You are analyzing a proof of payment / receipt for a Mexican insurance policy payment.
Extract ONLY the following fields as valid JSON, no markdown fences, no extra text:
{{
  "amount": "the paid amount with currency exactly as shown, or null",
  "insurance_id": "the policy or certificate number shown on the receipt, or null",
  "payment_date": "YYYY-MM-DD date the payment was made or is for, or null",
  "beneficiary": "payee or beneficiary name on the receipt, or null",
  "company": "the insurance company or payee name shown on the receipt, or null"
}}

Rules:
- {lang_instruction}
- CRITICAL: You MUST extract "amount" and "payment_date". Search the entire document carefully.
  Look for amount labels: monto, importe, total, cantidad, amount paid, payment amount.
  Look for date labels: fecha de pago, fecha, payment date, date paid, fecha de operación.
  Look for beneficiary labels: beneficiario, payee, a favor de, receptor.
- If a field is not visible on the document after careful search, use null.
- Return ONLY the JSON object, nothing else.

DOCUMENT TEXT:
{text[:8000]}
"""


def analyze_payment_proof_text(text: str, language: str) -> tuple[dict[str, Any] | None, str | None]:
    api_key = get_api_key()
    if not api_key:
        return None, "err_no_api_key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(get_gemini_model())
        response = model.generate_content(build_payment_proof_prompt(text, language))
        if not response.candidates:
            return None, "payment_extract_error"
        raw = response.text
        if not raw or not raw.strip():
            return None, "payment_extract_error"
        return parse_gemini_json(raw.strip()), None
    except Exception as exc:
        print(f"[analyze_payment_proof_text] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None, "payment_extract_error"


def analyze_payment_proof_image(file: Any, language: str) -> tuple[dict[str, Any] | None, str | None]:
    api_key = get_api_key()
    if not api_key:
        return None, "err_no_api_key"
    try:
        file.seek(0)
        image_bytes = file.read()
        mime = getattr(file, "type", None) or "image/jpeg"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(get_gemini_model())
        prompt = build_payment_proof_prompt("(scanned image — read the fields visually)", language)
        response = model.generate_content([{"mime_type": mime, "data": image_bytes}, prompt])
        if not response.candidates:
            return None, "payment_extract_error"
        raw = response.text
        if not raw or not raw.strip():
            return None, "payment_extract_error"
        return parse_gemini_json(raw.strip()), None
    except Exception as exc:
        print(f"[analyze_payment_proof_image] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None, "payment_extract_error"


def analyze_payment_proof(file: Any, language: str) -> tuple[dict[str, Any] | None, str | None]:
    name = (getattr(file, "name", "") or "").lower()
    if name.endswith(".pdf"):
        text, err, _detail = extract_text(file)
        if err or not text:
            return None, "payment_extract_error"
        return analyze_payment_proof_text(text, language)
    return analyze_payment_proof_image(file, language)


def normalize_amount_value(value: str) -> str:
    digits = re.sub(r"[^0-9.]", "", value or "")
    if not digits:
        return ""
    try:
        return f"{float(digits):.2f}"
    except ValueError:
        return digits


def normalize_company_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def payment_proof_amount_present(value: Any) -> bool:
    return bool(normalize_amount_value(str(value or "")))


def payment_proof_date_present(value: Any) -> bool:
    return parse_policy_date(str(value) if value is not None else None) is not None


def get_missing_payment_proof_fields(extracted: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not payment_proof_amount_present(extracted.get("amount")):
        missing.append("amount")
    if not payment_proof_date_present(extracted.get("payment_date")):
        missing.append("date")
    return missing


def merge_manual_payment_fields(
    extracted: dict[str, Any],
    values: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(extracted)
    amount = values.get("amount")
    if amount is not None and str(amount).strip():
        merged["amount"] = str(amount).strip()
    payment_date = values.get("payment_date")
    if isinstance(payment_date, date):
        merged["payment_date"] = payment_date.isoformat()
    elif payment_date is not None and str(payment_date).strip():
        parsed = parse_policy_date(str(payment_date))
        if parsed:
            merged["payment_date"] = parsed.isoformat()
        else:
            merged["payment_date"] = str(payment_date).strip()
    return merged


def get_current_term_due_iso(analysis: dict[str, Any], context: dict[str, Any] | None = None) -> str | None:
    if context is not None:
        due = get_next_unpaid_payment_date(analysis, context)
        if due:
            return due.isoformat()
    parsed = parse_policy_date(analysis.get("renewal_date"))
    return parsed.isoformat() if parsed else None


def get_payment_beneficiary(extracted: dict[str, Any]) -> str:
    value = extracted.get("beneficiary") or extracted.get("company")
    if value and str(value).strip().lower() not in {"null", "none", ""}:
        return str(value).strip()
    return t("not_found")


def is_current_term_paid(context: dict[str, Any], analysis: dict[str, Any]) -> bool:
    current_due = get_current_term_due_iso(analysis, context)
    if not current_due:
        return False
    if context.get("current_term_paid") and context.get("current_term_paid_due") == current_due:
        return True
    rec = (context.get("payment_confirmations") or {}).get(current_due)
    return bool(
        rec
        and rec.get("status") == "confirmed"
        and rec.get("method") == "auto"
        and is_term_paid_record(rec)
    )


def persist_analysis_context(analysis_id: int, context: dict[str, Any]) -> None:
    update_analysis_record(analysis_id, context_json=json.dumps(context))


def payment_date_in_valid_window(
    payment_date: date,
    due_date: date,
    extension_days: int = 0,
) -> bool:
    earliest = due_date - timedelta(days=30)
    latest = due_date + timedelta(days=max(0, int(extension_days)))
    return earliest <= payment_date <= latest


def format_payment_date_window(due_date: date, extension_days: int = 0) -> str:
    window_start = format_display_date(due_date - timedelta(days=30))
    window_end = format_display_date(due_date + timedelta(days=max(0, int(extension_days))))
    return f"{window_start} – {window_end}"


def get_extension_allowance_days(context: dict[str, Any]) -> int:
    try:
        return max(0, int(context.get("extension_allowance_days") or 0))
    except (TypeError, ValueError):
        return 0


def render_payment_extracted_summary(extracted: dict[str, Any]) -> None:
    st.markdown(f"**{t('payment_extracted_title')}**")
    st.markdown(
        f'<div class="payment-extracted-summary">'
        f"<strong>{t('payment_field_amount')}:</strong> {extracted.get('amount') or t('not_found')}<br>"
        f"<strong>{t('payment_field_date')}:</strong> {format_policy_date(extracted.get('payment_date'))}<br>"
        f"<strong>{t('payment_field_beneficiary')}:</strong> {get_payment_beneficiary(extracted)}"
        f"</div>",
        unsafe_allow_html=True,
    )


def match_payment_proof(
    extracted: dict[str, Any],
    expected_amount: str,
    expected_insurance_id: str,
    expected_date: date,
    expected_company: str,
    extension_days: int = 0,
) -> dict[str, Any]:
    ext_amount = normalize_amount_value(str(extracted.get("amount") or ""))
    exp_amount = normalize_amount_value(expected_amount)
    amount_match = bool(ext_amount) and bool(exp_amount) and ext_amount == exp_amount

    date_match = False
    ext_date = parse_policy_date(str(extracted.get("payment_date") or ""))
    if ext_date:
        date_match = payment_date_in_valid_window(ext_date, expected_date, extension_days)

    matches = {
        "amount": amount_match,
        "date": date_match,
    }
    return {"matches": matches, "all_match": amount_match and date_match}


def render_extension_allowance_controls(
    key_prefix: str,
    context: dict[str, Any],
    analysis_id: int | None,
) -> None:
    ext_key = f"{key_prefix}extension_allowance"
    if ext_key not in st.session_state:
        st.session_state[ext_key] = get_extension_allowance_days(context)

    ext_col, save_col = st.columns([2.2, 0.75], gap="small", vertical_alignment="bottom")
    with ext_col:
        st.number_input(
            t("extension_allowance_label"),
            min_value=0,
            max_value=365,
            step=1,
            key=ext_key,
            help=t("extension_allowance_help"),
        )
    with save_col:
        st.markdown('<div class="save-btn-blue-marker"></div>', unsafe_allow_html=True)
        if st.button(t("save_btn"), key=f"{key_prefix}save_extension_allowance", width="stretch"):
            allowance = int(st.session_state.get(ext_key, 0))
            context["extension_allowance_days"] = max(0, allowance)
            if analysis_id:
                persist_analysis_context(int(analysis_id), context)
                if st.session_state.get("saved_analysis_id") == analysis_id:
                    st.session_state.analysis_context = context
            st.success(t("extension_allowance_saved"))


def compute_risk_level(
    renewal_date_str: str | None,
    reference: date | None = None,
    *,
    alert_days: int | None = None,
    red_flag_days: int | None = None,
) -> tuple[str, str]:
    if not renewal_date_str:
        return "warning", "err_no_renewal"
    try:
        renewal = datetime.strptime(renewal_date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return "warning", "err_no_renewal"
    ref = reference or get_reference_date()
    days_until = (renewal - ref).days
    alert = int(alert_days if alert_days is not None else get_renewal_alert_days())
    red_flag = int(red_flag_days if red_flag_days is not None else get_renewal_red_flag_days())
    if red_flag > alert:
        red_flag = alert
    if days_until <= red_flag:
        return "critical", "risk_within_7"
    if days_until <= alert:
        return "warning", "risk_within_30"
    return "ok", "risk_ok"


def payments_per_year(freq: str) -> int:
    return {"monthly": 12, "quarterly": 4, "semester": 2, "annual": 1}.get(freq, 1)


def months_between_payments(freq: str) -> int:
    return {"monthly": 1, "quarterly": 3, "semester": 6, "annual": 12}.get(freq, 12)


def add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    days_in_month = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                     31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return date(year, month, min(value.day, days_in_month))


def generate_policy_payment_dates(
    next_due: date,
    freq: str,
    policy_start: date | None = None,
    policy_end: date | None = None,
) -> list[date]:
    """Build payment dates for the current policy term."""
    count = payments_per_year(freq)
    interval = months_between_payments(freq)

    if policy_start:
        dates: list[date] = []
        current = policy_start
        while len(dates) < count:
            if policy_end and current > policy_end:
                break
            dates.append(current)
            current = add_months(current, interval)
        if dates:
            return dates

    dates = [next_due]
    current = next_due
    for _ in range(count - 1):
        current = add_months(current, interval)
        if policy_end and current > policy_end:
            break
        dates.append(current)
    return dates


def get_policy_payment_schedule(
    analysis: dict[str, Any],
    context: dict[str, Any],
) -> list[date]:
    renewal = analysis.get("renewal_date") or ""
    if not renewal or renewal == "null":
        return []
    try:
        anchor_due = datetime.strptime(renewal[:10], "%Y-%m-%d").date()
    except ValueError:
        return []
    freq = normalize_payment_frequency(
        context.get("payment_frequency") or analysis.get("payment_frequency")
    )
    policy_start = parse_policy_date(analysis.get("policy_start_date"))
    policy_end = parse_policy_date(analysis.get("policy_end_date"))
    return generate_policy_payment_dates(anchor_due, freq, policy_start, policy_end)


def get_next_unpaid_payment_date(
    analysis: dict[str, Any],
    context: dict[str, Any],
    reference: date | None = None,
) -> date | None:
    reference = reference or get_reference_date()
    payment_dates = get_policy_payment_schedule(analysis, context)
    confirmations = context.get("payment_confirmations") or {}
    if payment_dates:
        for due_date in payment_dates:
            if not is_term_paid(confirmations, due_date.isoformat()):
                return due_date
        return None
    return parse_policy_date(analysis.get("renewal_date"))


def get_reminder_premium_amount(analysis: dict[str, Any], context: dict[str, Any]) -> str:
    premium_total = analysis.get("premium_amount") or t("not_found")
    freq = normalize_payment_frequency(
        context.get("payment_frequency") or analysis.get("payment_frequency")
    )
    return compute_premium_per_term(premium_total, freq)


def get_reminder_due_context(
    analysis: dict[str, Any],
    context: dict[str, Any],
    reference: date | None = None,
) -> tuple[str | None, date | None, str]:
    """Return (due_date_str, due_date, premium_per_term) for outbound reminders."""
    due = get_next_unpaid_payment_date(analysis, context, reference)
    premium = get_reminder_premium_amount(analysis, context)
    if due:
        return due.isoformat(), due, premium
    renewal = analysis.get("renewal_date") or ""
    if renewal and renewal != "null":
        parsed = parse_policy_date(renewal)
        return renewal[:10], parsed, premium
    return None, None, premium


def sanitize_reminder_rationale(rationale: str, premium_amount: str = "") -> str:
    if not rationale:
        return ""
    text = rationale.strip()
    if premium_amount and premium_amount != t("not_found"):
        text = text.replace(premium_amount, "")
        amount = extract_first_amount(premium_amount)
        if amount is not None:
            text = re.sub(re.escape(f"{amount:.2f}"), "", text)
            text = re.sub(re.escape(f"{amount:,.2f}"), "", text)
            text = re.sub(re.escape(str(amount)), "", text)
    text = re.sub(r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:MXN|USD|pesos?))?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:MXN|USD|pesos?)\s*[\d,]+(?:\.\d{2})?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:premium|prima)\s*:?\s*[\d,]+(?:\.\d{2})?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\.\s*\.", ".", text)
    return text.strip(" ,.;")


def payment_due_color(due: date, reference: date) -> str:
    days = (due - reference).days
    if days > 30:
        return "green"
    if days > 7:
        return "yellow"
    return "red"


def format_payment_status_text(due: date, reference: date) -> str:
    days = (due - reference).days
    if days < 0:
        return t("payment_days_overdue").format(days=abs(days))
    if days == 0:
        return t("payment_due_today")
    return t("payment_days_left").format(days=days)


def is_term_paid_record(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    if record.get("paid") is True:
        return True
    return record.get("status") == "confirmed"


def is_term_paid(confirmations: dict[str, Any], due_iso: str) -> bool:
    return is_term_paid_record(confirmations.get(due_iso))


def format_payment_summary_status(
    due_date: date,
    reference: date,
    confirmation: dict[str, Any] | None,
) -> str:
    if is_term_paid_record(confirmation):
        return t("payment_status_paid")
    return format_payment_status_text(due_date, reference)


def payment_status_label(color: str) -> str:
    return {
        "green": t("payment_status_ok"),
        "yellow": t("payment_status_warning"),
        "red": t("payment_status_critical"),
    }.get(color, t("payment_status_ok"))


def get_history_renewal_status(
    analysis: dict[str, Any],
    reference: date | None = None,
) -> tuple[str, str]:
    if not policy_term_dates_available(analysis):
        return "red", t("renewal_no_dates")
    renewal_due = get_policy_renewal_due_value(analysis)
    if not renewal_due:
        return "red", t("renewal_no_dates")
    risk_level, _ = compute_risk_level(renewal_due, reference)
    mapping = {
        "ok": ("green", "renewal_status_good"),
        "warning": ("yellow", "renewal_status_renew"),
        "critical": ("red", "renewal_status_urgent"),
    }
    color, text_key = mapping.get(risk_level, mapping["ok"])
    return color, t(text_key)


def get_history_payment_status(
    analysis: dict[str, Any],
    context: dict[str, Any],
    reference: date | None = None,
) -> tuple[str, str]:
    ref = reference or get_reference_date()
    renewal = analysis.get("renewal_date") or ""
    if not renewal or renewal == "null":
        return "yellow", t("payment_status_warning")
    try:
        due = datetime.strptime(renewal[:10], "%Y-%m-%d").date()
    except ValueError:
        return "yellow", t("payment_status_warning")

    confirmations = context.get("payment_confirmations") or {}
    confirmed = confirmations.get(due.isoformat())
    if is_term_paid_record(confirmed):
        return "green", t("payment_status_paid")

    color = payment_due_color(due, ref)
    return color, payment_status_label(color)


def render_history_status_chip(color: str, text: str) -> None:
    css_class = {
        "green": "history-chip-green",
        "yellow": "history-chip-yellow",
        "red": "history-chip-red",
    }.get(color, "history-chip-green")
    safe_text = xml.sax.saxutils.escape(text)
    st.markdown(
        f'<div class="history-status-chip {css_class}">{safe_text}</div>',
        unsafe_allow_html=True,
    )


def render_payment_proof_missing_form(
    *,
    pending_key: str,
    verify_key: str,
    upload_reset_key: str,
    uploader_key: str,
    due_iso: str,
    due_date: date,
    expected_amount: str,
    expected_insurance_id: str,
    expected_company: str,
    context: dict[str, Any],
    analysis_id: int,
    lang: str,
) -> None:
    pending = st.session_state.get(pending_key) or {}
    extracted = pending.get("extracted") or {}
    missing_keys: list[str] = pending.get("missing") or []
    if not missing_keys:
        return

    st.warning(t("payment_missing_title"))
    st.caption(t("payment_missing_hint"))
    render_payment_extracted_summary(extracted)
    payment_proof_labels = {
        "amount": t("payment_field_amount"),
        "date": t("payment_field_date"),
    }
    for field_key in missing_keys:
        st.markdown(
            t("upload_missing_field_item").format(field=payment_proof_labels.get(field_key, field_key))
        )

    with st.form(f"payment_missing_{due_iso}"):
        inputs: dict[str, Any] = {}
        if "amount" in missing_keys:
            inputs["amount"] = st.text_input(t("payment_field_amount"))
        if "date" in missing_keys:
            inputs["payment_date"] = st.date_input(t("payment_field_date"), value=due_date)
        col_continue, col_retry = st.columns(2)
        with col_continue:
            continue_btn = st.form_submit_button(t("payment_manual_continue"), type="primary")
        with col_retry:
            retry_btn = st.form_submit_button(t("payment_try_again"))

    if retry_btn:
        st.session_state.pop(pending_key, None)
        st.session_state.pop(verify_key, None)
        st.session_state[upload_reset_key] = st.session_state.get(upload_reset_key, 0) + 1
        st.rerun()

    if continue_btn:
        merged = merge_manual_payment_fields(extracted, inputs)
        still_missing = get_missing_payment_proof_fields(merged)
        if still_missing:
            st.session_state[pending_key] = {"extracted": merged, "missing": still_missing}
            st.error(t("payment_missing_title"))
            st.rerun()
            return
        st.session_state.pop(pending_key, None)
        extension_days = get_extension_allowance_days(context)
        result = match_payment_proof(
            merged,
            expected_amount,
            expected_insurance_id,
            due_date,
            expected_company,
            extension_days,
        )
        st.session_state[verify_key] = {"extracted": merged, **result}
        st.rerun()


def render_payment_verification(
    *,
    key_prefix: str,
    due_iso: str,
    due_date: date,
    analysis_id: int,
    analysis: dict[str, Any],
    expected_amount: str,
    expected_insurance_id: str,
    expected_company: str,
    lang: str,
    context: dict[str, Any],
) -> None:
    verify_key = f"{key_prefix}pay_verify_{due_iso}"
    override_key = f"{key_prefix}pay_override_{due_iso}"
    pending_key = f"{key_prefix}pay_pending_{due_iso}"
    upload_reset_key = f"{key_prefix}pay_upload_reset_{due_iso}"
    if upload_reset_key not in st.session_state:
        st.session_state[upload_reset_key] = 0
    uploader_key = f"{key_prefix}pay_upload_{due_iso}_{st.session_state[upload_reset_key]}"
    extension_days = get_extension_allowance_days(context)

    def _persist_confirmation(record: dict[str, Any]) -> None:
        save_payment_confirmation(analysis_id, due_iso, record)
        confirmations = context.setdefault("payment_confirmations", {})
        confirmations[due_iso] = record
        current_due = get_current_term_due_iso(analysis, context)
        if (
            record.get("method") == "auto"
            and current_due
            and due_iso == current_due
            and record.get("matches", {}).get("amount")
            and record.get("matches", {}).get("date")
        ):
            context["current_term_paid"] = True
            context["current_term_paid_due"] = current_due
            persist_analysis_context(analysis_id, context)
        if st.session_state.get("saved_analysis_id") == analysis_id:
            st.session_state.analysis_context = context

    if st.session_state.get(pending_key):
        render_payment_proof_missing_form(
            pending_key=pending_key,
            verify_key=verify_key,
            upload_reset_key=upload_reset_key,
            uploader_key=uploader_key,
            due_iso=due_iso,
            due_date=due_date,
            expected_amount=expected_amount,
            expected_insurance_id=expected_insurance_id,
            expected_company=expected_company,
            context=context,
            analysis_id=analysis_id,
            lang=lang,
        )
        return

    verify_result = st.session_state.get(verify_key)

    if verify_result:
        render_payment_extracted_summary(verify_result.get("extracted") or {})

    if verify_result and verify_result.get("all_match"):
        _persist_confirmation({
            "status": "confirmed",
            "paid": True,
            "due_date": due_iso,
            "confirmed_at": datetime.now().isoformat(),
            "method": "auto",
            "extracted": verify_result["extracted"],
            "matches": verify_result["matches"],
        })
        st.session_state.pop(verify_key, None)
        st.success(t("payment_confirmed_auto"))
        st.rerun()
        return

    if verify_result and not verify_result.get("all_match"):
        extracted = verify_result["extracted"]
        matches = verify_result["matches"]
        st.warning(t("payment_mismatch_title"))
        field_specs = [
            ("amount", t("payment_field_amount"), expected_amount, extracted.get("amount")),
            (
                "date",
                t("payment_field_date"),
                format_payment_date_window(due_date, extension_days),
                extracted.get("payment_date"),
            ),
        ]
        for field_key, label, expected_val, found_val in field_specs:
            ok = matches.get(field_key, False)
            css = "payment-match-ok" if ok else "payment-match-fail"
            match_text = t("payment_match_ok") if ok else t("payment_match_fail")
            found_display = found_val or t("not_found")
            st.markdown(
                f'<div class="payment-match-row {css}">'
                f"<strong>{label}:</strong> {t('payment_expected_label')} \"{expected_val}\" · "
                f"{t('payment_found_label')} \"{found_display}\" — {match_text}"
                f"</div>",
                unsafe_allow_html=True,
            )

        col_retry, col_override = st.columns(2)
        with col_retry:
            if st.button(t("payment_mismatch_reupload_btn"), key=f"{key_prefix}pay_retry_{due_iso}"):
                st.session_state.pop(verify_key, None)
                st.session_state.pop(override_key, None)
                st.session_state.pop(pending_key, None)
                st.session_state[upload_reset_key] = st.session_state.get(upload_reset_key, 0) + 1
                st.rerun()
        with col_override:
            if st.button(t("payment_mismatch_override_btn"), key=f"{key_prefix}pay_override_btn_{due_iso}"):
                st.session_state[override_key] = True

        if st.session_state.get(override_key):
            st.error(t("payment_mismatch_override_warning"))
            if st.button(
                t("payment_mismatch_confirm_btn"),
                key=f"{key_prefix}pay_confirm_{due_iso}",
                type="primary",
            ):
                _persist_confirmation({
                    "status": "confirmed",
                    "paid": True,
                    "due_date": due_iso,
                    "confirmed_at": datetime.now().isoformat(),
                    "method": "manual_override",
                    "extracted": extracted,
                    "matches": matches,
                })
                st.session_state.pop(verify_key, None)
                st.session_state.pop(override_key, None)
                st.success(t("payment_confirmed_override"))
                st.rerun()
        return

    uploaded = st.file_uploader(
        t("payment_upload_title"),
        type=["pdf", "png", "jpg", "jpeg"],
        help=t("payment_upload_help"),
        key=uploader_key,
        accept_multiple_files=False,
    )

    extract_cache_key = f"{key_prefix}pay_extracted_{due_iso}_{st.session_state[upload_reset_key]}"
    extract_file_key = f"{extract_cache_key}_file"
    cached_extracted = st.session_state.get(extract_cache_key)

    if uploaded is not None:
        file_signature = f"{uploaded.name}:{getattr(uploaded, 'size', '')}"
        if st.session_state.get(extract_file_key) != file_signature:
            with st.spinner(t("payment_verifying")):
                extracted, err = analyze_payment_proof(uploaded, lang)
            if err or extracted is None:
                st.error(t("payment_extract_error"))
                st.session_state.pop(extract_cache_key, None)
                st.session_state.pop(extract_file_key, None)
            else:
                st.session_state[extract_cache_key] = extracted
                st.session_state[extract_file_key] = file_signature
                cached_extracted = extracted
        if cached_extracted and not verify_result:
            render_payment_extracted_summary(cached_extracted)

    if uploaded is not None and verify_result is None and cached_extracted:
        if st.button(t("payment_verify_btn"), key=f"{key_prefix}pay_verify_btn_{due_iso}", type="primary"):
            extracted = cached_extracted
            missing = get_missing_payment_proof_fields(extracted)
            if missing:
                st.session_state[pending_key] = {
                    "extracted": extracted,
                    "missing": missing,
                }
                st.rerun()
            else:
                result = match_payment_proof(
                    extracted,
                    expected_amount,
                    expected_insurance_id,
                    due_date,
                    expected_company,
                    extension_days,
                )
                st.session_state[verify_key] = {"extracted": extracted, **result}
                st.rerun()


def render_payment_subrecords(
    analysis: dict[str, Any],
    context: dict[str, Any],
    *,
    key_prefix: str = "",
) -> None:
    renewal = analysis.get("renewal_date") or ""
    if not renewal or renewal == "null":
        return
    try:
        next_due = datetime.strptime(renewal[:10], "%Y-%m-%d").date()
    except ValueError:
        return

    freq = normalize_payment_frequency(
        context.get("payment_frequency") or analysis.get("payment_frequency")
    )
    policy_start = parse_policy_date(analysis.get("policy_start_date"))
    policy_end = parse_policy_date(analysis.get("policy_end_date"))
    payment_dates = generate_policy_payment_dates(next_due, freq, policy_start, policy_end)
    reference = get_reference_date()
    premium_total = analysis.get("premium_amount", t("not_found"))
    premium_per_term = compute_premium_per_term(premium_total, freq)
    total = len(payment_dates)
    analysis_id = context.get("analysis_id")
    insurance_id_expected = analysis.get("policy_number") or context.get("insurance_id") or get_insurance_id(analysis)
    insurer_expected = analysis.get("insurer", "")
    lang = context.get("analysis_language") or st.session_state.language
    confirmations = context.get("payment_confirmations") or {}

    with st.container():
        st.markdown(f'<div class="payment-schedule-marker"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-heading">{t("payment_schedule_title")}</div>', unsafe_allow_html=True)
        render_extension_allowance_controls(key_prefix, context, analysis_id)
        st.caption(freq_label(freq))

        for index, due_date in enumerate(payment_dates, start=1):
            due_iso = due_date.isoformat()
            confirmed = confirmations.get(due_iso)
            term_paid = is_term_paid_record(confirmed)
            color = "green" if term_paid else payment_due_color(due_date, reference)
            status_text = format_payment_summary_status(due_date, reference, confirmed)
            summary = t("payment_item_summary").format(
                n=index,
                total=total,
                date=format_display_date(due_date),
                status=status_text,
            )

            st.markdown(f'<div class="payment-marker payment-marker-{color}"></div>', unsafe_allow_html=True)
            with st.expander(summary, expanded=False):
                if term_paid:
                    confirmed_at_raw = (confirmed.get("confirmed_at") or "")[:10]
                    confirmed_display = format_policy_date(confirmed_at_raw) if confirmed_at_raw else t("not_found")
                    st.markdown(
                        f'<span class="payment-paid-badge">'
                        f'{t("payment_confirmed_badge").format(date=confirmed_display)}'
                        f"</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="payment-body">'
                        f"<strong>{t('payment_detail_due')}:</strong> {format_display_date(due_date)}<br>"
                        f"<strong>{t('premium_per_term_label')}:</strong> {premium_per_term}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="payment-body">'
                        f"<strong>{t('payment_detail_due')}:</strong> {format_display_date(due_date)}<br>"
                        f"<strong>{t('payment_detail_status')}:</strong> {payment_status_label(color)}<br>"
                        f"<strong>{t('premium_per_term_label')}:</strong> {premium_per_term}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
                    if analysis_id:
                        render_payment_verification(
                            key_prefix=key_prefix,
                            due_iso=due_iso,
                            due_date=due_date,
                            analysis_id=int(analysis_id),
                            analysis=analysis,
                            expected_amount=premium_per_term,
                            expected_insurance_id=insurance_id_expected,
                            expected_company=insurer_expected,
                            lang=lang,
                            context=context,
                        )
                    else:
                        st.caption(t("payment_save_first_hint"))


def reminders_on_date(
    due_date: date,
    ref_date: date,
    start_days: int,
    base_time: str,
    frequent_start_days: int = 7,
    daily_frequency: int = 2,
) -> list[dict[str, Any]]:
    schedule = build_reminder_schedule(due_date, start_days, base_time, frequent_start_days, daily_frequency)
    return [r for r in schedule if r["date"] == ref_date.isoformat()]


def dispatch_reminders_for_record(
    analysis: dict[str, Any],
    ref_date: date,
    lang: str,
    config: dict[str, str],
    *,
    email_settings: dict[str, Any] | None = None,
    whatsapp_settings: dict[str, Any] | None = None,
    payment_confirmations: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[list[str], bool]:
    """Send configured channels when a reminder falls on ref_date. Returns (messages, reminder_was_due)."""
    email_settings = email_settings or {}
    whatsapp_settings = whatsapp_settings or {}
    reminder_context = dict(context or {})
    if payment_confirmations is not None:
        reminder_context["payment_confirmations"] = payment_confirmations
    reminder_context.setdefault("payment_frequency", analysis.get("payment_frequency"))

    rd, due, premium = get_reminder_due_context(analysis, reminder_context, ref_date)
    if not rd or not due:
        return [], False

    confirmations = reminder_context.get("payment_confirmations") or {}
    if is_term_paid(confirmations, due.isoformat()):
        return [], False

    email_days = int(email_settings.get("reminder_start_days") or 30)
    email_time = email_settings.get("reminder_base_time") or "09:00"
    email_frequent_days = int(email_settings.get("frequent_start_days") or 7)
    email_daily_freq = int(email_settings.get("daily_frequency") or 2)
    wa_days = int(whatsapp_settings.get("reminder_start_days") or 30)
    wa_time = whatsapp_settings.get("reminder_base_time") or "09:00"
    wa_frequent_days = int(whatsapp_settings.get("frequent_start_days") or 7)
    wa_daily_freq = int(whatsapp_settings.get("daily_frequency") or 2)
    email_due_rems = reminders_on_date(due, ref_date, email_days, email_time, email_frequent_days, email_daily_freq)
    wa_due_rems = reminders_on_date(due, ref_date, wa_days, wa_time, wa_frequent_days, wa_daily_freq)
    if not email_due_rems and not wa_due_rems:
        return [], False

    holder = analysis.get("policy_holder", "")
    emails = [
        (email_settings.get("primary") or "").strip(),
        (email_settings.get("secondary") or "").strip(),
    ]
    phones = [
        (whatsapp_settings.get("primary") or "").strip(),
        (whatsapp_settings.get("secondary") or "").strip(),
    ]
    sent: list[str] = []

    if email_due_rems and emails[0] and is_smtp_configured(config):
        if send_email_reminder(emails, holder, rd, premium, lang):
            sent.append(t("email_sent"))

    phone = phones[0]
    if wa_due_rems and phone and is_voice_configured(config):
        ok, _ = send_voice_reminder(phone, holder, rd, premium, lang)
        if ok:
            sent.append(t("voice_sent"))

    if wa_due_rems and phone and is_sms_configured(config):
        ok, _ = send_sms_reminder(phone, holder, rd, premium, lang)
        if ok:
            sent.append(t("sms_sent"))

    if wa_due_rems and phone and is_twilio_configured(config):
        ok, _ = send_whatsapp_reminder(phone, holder, rd, premium, lang)
        if ok:
            sent.append(t("whatsapp_sent"))

    if not sent:
        return [t("sim_no_contact")], True
    return sent, True


def run_simulation_for_all_records(sim_date: date, config: dict[str, str]) -> list[dict[str, Any]]:
    """Evaluate every saved policy and send reminders due on sim_date."""
    resolved = resolve_config(config)
    results: list[dict[str, Any]] = []
    history = get_history()
    if not history:
        return results

    for row in history:
        analysis, context = parse_stored_analysis(row)
        holder = analysis.get("policy_holder") or t("not_found")

        if not row.get("reminders_active", 1):
            results.append({
                "summary": t("sim_record_inactive").format(holder=holder),
                "sent_count": 0,
            })
            continue

        rd = analysis.get("renewal_date") or row.get("renewal_date") or ""
        if not rd or rd == "null":
            results.append({
                "summary": t("sim_record_no_renewal").format(holder=holder),
                "sent_count": 0,
            })
            continue

        email_cfg = get_channel_reminder_config("email", row, context, analysis)
        wa_cfg = get_channel_reminder_config("whatsapp", row, context, analysis)
        lang = context.get("analysis_language") or row.get("language") or st.session_state.language
        confirmations = load_payment_confirmations(row)

        messages, due = dispatch_reminders_for_record(
            analysis,
            sim_date,
            lang,
            resolved,
            email_settings=email_cfg,
            whatsapp_settings=wa_cfg,
            payment_confirmations=confirmations,
            context=context,
        )

        if not due:
            results.append({
                "summary": t("sim_record_not_due").format(holder=holder),
                "sent_count": 0,
            })
            continue

        ok_messages = [m for m in messages if m not in (t("sim_no_contact"), t("sim_no_alerts"))]
        if ok_messages:
            channels = ", ".join(ok_messages)
            results.append({
                "summary": t("sim_record_due").format(
                    holder=holder,
                    count=len(ok_messages),
                    channels=channels,
                ),
                "sent_count": len(ok_messages),
            })
        elif t("sim_no_contact") in messages:
            results.append({
                "summary": f"**{holder}** — {t('sim_no_contact')}",
                "sent_count": 0,
            })
        else:
            results.append({
                "summary": t("sim_record_not_due").format(holder=holder),
                "sent_count": 0,
            })

    return results


def normalize_payment_frequency(freq: str | None) -> str:
    if not freq:
        return "annual"
    freq = freq.lower().strip()
    mapping = {
        "mensual": "monthly", "monthly": "monthly",
        "trimestral": "quarterly", "quarterly": "quarterly",
        "semestral": "semester", "semester": "semester", "semi-annual": "semester",
        "anual": "annual", "annual": "annual", "yearly": "annual",
    }
    return mapping.get(freq, "annual")


def build_reminder_schedule(
    due_date: date,
    start_days: int,
    base_time: str = "09:00",
    frequent_start_days: int = 7,
    daily_frequency: int = 2,
) -> list[dict[str, Any]]:
    """
    Build escalating reminder schedule:
    - Normal (before the "frequent" window): once daily at base_time
    - Frequent (within frequent_start_days of the due date): `daily_frequency` times daily
    """
    frequent_start_days = min(int(frequent_start_days or 0), max(0, int(start_days) - 1))
    frequent_times = generate_daily_times(base_time, daily_frequency)
    reminders: list[dict[str, Any]] = []
    start_date = due_date - timedelta(days=start_days)
    current = start_date
    while current <= due_date:
        days_left = (due_date - current).days
        if days_left <= frequent_start_days:
            times, tier = frequent_times, "frequent"
        else:
            times, tier = [base_time], "normal"
        for time_str in times:
            reminders.append({
                "date": current.isoformat(),
                "time": time_str,
                "tier": tier,
                "count": len(times),
                "days_until_due": days_left,
            })
        current += timedelta(days=1)
    return reminders


def generate_ics_schedule(
    reminders: list[dict[str, Any]],
    policy_holder: str,
    language: str = "es",
) -> bytes:
    """Generate an ICS file with all scheduled reminder events."""
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    events: list[str] = []
    for i, rem in enumerate(reminders):
        uid = str(uuid.uuid4())
        dt = datetime.strptime(f"{rem['date']} {rem['time']}", "%Y-%m-%d %H:%M")
        dt_str = dt.strftime("%Y%m%dT%H%M%S")
        if language == "en":
            summary = f"GMM Payment Reminder — {policy_holder}"
            desc = f"Payment due in {rem['days_until_due']} days. Tier: {rem['tier']}"
        else:
            summary = f"Recordatorio de Pago GMM — {policy_holder}"
            desc = f"Pago en {rem['days_until_due']} días. Nivel: {rem['tier']}"
        events.append(f"""BEGIN:VEVENT
UID:{uid}
DTSTAMP:{now}
DTSTART:{dt_str}
SUMMARY:{summary}
DESCRIPTION:{desc}
STATUS:CONFIRMED
BEGIN:VALARM
TRIGGER:-PT0M
ACTION:DISPLAY
DESCRIPTION:Payment reminder
END:VALARM
END:VEVENT""")

    body = "\n".join(events)
    return f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Insurance Renewal Guardian v2//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
{body}
END:VCALENDAR
""".encode("utf-8")


def build_reminder_message(
    policy_holder: str,
    renewal_date_str: str,
    premium: str,
    language: str = "es",
) -> tuple[str, str]:
    """Build email/WhatsApp subject and body for a renewal reminder."""
    if language == "en":
        subject = f"GMM payment reminder — {policy_holder}"
        body = (
            f"Hello {policy_holder},\n\n"
            f"Your Gastos Médicos Mayores policy payment is due on {renewal_date_str}.\n"
            f"Premium: {premium}\n\n"
            f"Contact your agent to confirm payment and avoid losing coverage.\n\n"
            f"— Insurance Renewal Guardian"
        )
    else:
        subject = f"Recordatorio de pago GMM — {policy_holder}"
        body = (
            f"Hola {policy_holder},\n\n"
            f"El pago de tu póliza de Gastos Médicos Mayores vence el {renewal_date_str}.\n"
            f"Prima: {premium}\n\n"
            f"Contacta a tu agente para confirmar el pago y no perder tu cobertura.\n\n"
            f"— Guardián de Renovación de Seguros"
        )
    return subject, body


TWILIO_TRIAL_SMS_MAX = 160


def gsm_safe_text(text: str) -> str:
    """Strip characters that force UCS-2 encoding and shrink trial SMS segments."""
    replacements = {
        "—": "-", "–": "-", "…": "...", "“": '"', "”": '"', "‘": "'", "’": "'",
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ü": "U", "Ñ": "N",
    }
    cleaned = text
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    return cleaned.encode("ascii", "ignore").decode("ascii")


def build_twilio_reminder_body(
    policy_holder: str,
    renewal_date_str: str,
    premium: str,
    language: str = "es",
) -> str:
    """Compact GSM-safe body for Twilio SMS/WhatsApp (trial = 1 segment max)."""
    parsed = parse_policy_date(renewal_date_str)
    short_date = format_display_date(parsed) if parsed else renewal_date_str[:10]
    prem = re.sub(r"\s+", " ", (premium or "").strip())
    if len(prem) > 36:
        prem = prem[:33].rstrip() + "..."

    if language == "en":
        body = f"GMM payment due {short_date}. Premium: {prem}."
    else:
        body = f"Pago GMM vence {short_date}. Prima: {prem}."

    body = gsm_safe_text(body)
    if len(body) > TWILIO_TRIAL_SMS_MAX:
        if language == "en":
            body = f"GMM payment due {short_date}. Contact your agent."
        else:
            body = f"Pago GMM vence {short_date}. Contacta a tu agente."
        body = gsm_safe_text(body)
    return body[:TWILIO_TRIAL_SMS_MAX]


def clean_phone_e164(phone: str) -> str:
    """Strip to +digits only — never changes digits (must match Twilio exactly)."""
    raw = phone.strip()
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("00"):
        digits = digits[2:]
    if digits:
        return f"+{digits}"
    return raw


def build_whatsapp_deeplink(phone: str, message: str) -> str:
    """WhatsApp click-to-chat link — works on any phone without Twilio sandbox."""
    normalized = clean_phone_e164(phone)
    digits = re.sub(r"\D", "", normalized)
    return f"https://wa.me/{digits}?text={quote(message)}"


def send_email_reminder(
    to_addresses: list[str],
    policy_holder: str,
    renewal_date_str: str,
    premium: str,
    language: str = "es",
) -> bool:
    """Send a renewal reminder email via SMTP settings in .env."""
    config = load_config()
    if not is_smtp_configured(config):
        return False

    recipients = [e.strip() for e in to_addresses if e and e.strip()]
    if not recipients:
        return False

    subject, body = build_reminder_message(policy_holder, renewal_date_str, premium, language)
    try:
        msg = MIMEMultipart()
        msg["From"] = config["SMTP_FROM"]
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        port = int(config.get("SMTP_PORT") or ("465" if smtp_use_ssl(config) else "587"))
        if smtp_use_ssl(config):
            with smtplib.SMTP_SSL(config["SMTP_HOST"], port, timeout=30) as server:
                server.login(config["SMTP_USER"], config["SMTP_PASSWORD"])
                server.sendmail(config["SMTP_FROM"], recipients, msg.as_string())
        else:
            with smtplib.SMTP(config["SMTP_HOST"], port, timeout=30) as server:
                server.starttls()
                server.login(config["SMTP_USER"], config["SMTP_PASSWORD"])
                server.sendmail(config["SMTP_FROM"], recipients, msg.as_string())
        return True
    except Exception as exc:
        print(f"[send_email_reminder] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def send_whatsapp_reminder(
    phone: str,
    policy_holder: str,
    renewal_date_str: str,
    premium: str,
    language: str = "es",
) -> tuple[bool, str]:
    """Send a WhatsApp renewal reminder through Twilio. Returns (ok, detail_for_ui)."""
    config = resolve_config(load_config())
    if not is_twilio_configured(config):
        return False, "Twilio not configured"

    phone = clean_phone_e164(phone.strip())
    if not phone or phone == "+":
        return False, "Phone number empty"

    body = build_twilio_reminder_body(policy_holder, renewal_date_str, premium, language)
    to_number = f"whatsapp:{phone}"
    from_number = config["TWILIO_WHATSAPP_FROM"]

    try:
        from twilio.rest import Client

        client = Client(config["TWILIO_ACCOUNT_SID"], config["TWILIO_AUTH_TOKEN"])
        message = client.messages.create(from_=from_number, to=to_number, body=body)
        detail = (
            f"SID={message.sid} status={message.status} to={to_number} from={from_number}"
        )
        print(f"[send_whatsapp_reminder] {detail}")
        if message.error_code:
            err = f"{message.error_code}: {message.error_message}"
            print(f"[send_whatsapp_reminder] delivery error {err}")
            return False, err
        return True, f"{message.status}|{message.sid}"
    except Exception as exc:
        try:
            from twilio.base.exceptions import TwilioRestException

            if isinstance(exc, TwilioRestException):
                detail = f"{exc.code} — {exc.msg}"
            else:
                detail = f"{type(exc).__name__}: {exc}"
        except ImportError:
            detail = f"{type(exc).__name__}: {exc}"
        print(f"[send_whatsapp_reminder] {detail}")
        traceback.print_exc()
        return False, detail


def send_sms_reminder(
    phone: str,
    policy_holder: str,
    renewal_date_str: str,
    premium: str,
    language: str = "es",
) -> tuple[bool, str]:
    """Send a renewal reminder SMS through Twilio. Returns (ok, detail_for_ui)."""
    config = resolve_config(load_config())
    if not is_sms_configured(config):
        return False, "SMS not configured"

    phone = clean_phone_e164(phone.strip())
    if not phone or phone == "+":
        return False, "Phone number empty"

    body = build_twilio_reminder_body(policy_holder, renewal_date_str, premium, language)
    from_number = config["TWILIO_SMS_FROM"]

    try:
        from twilio.rest import Client

        client = Client(config["TWILIO_ACCOUNT_SID"], config["TWILIO_AUTH_TOKEN"])
        message = client.messages.create(from_=from_number, to=phone, body=body)
        detail = f"SID={message.sid} status={message.status} to={phone} from={from_number}"
        print(f"[send_sms_reminder] {detail}")
        if message.error_code:
            err = f"{message.error_code}: {message.error_message}"
            print(f"[send_sms_reminder] delivery error {err}")
            return False, err
        return True, f"{message.status}|{message.sid}"
    except Exception as exc:
        try:
            from twilio.base.exceptions import TwilioRestException

            if isinstance(exc, TwilioRestException):
                detail = f"{exc.code} — {exc.msg}"
            else:
                detail = f"{type(exc).__name__}: {exc}"
        except ImportError:
            detail = f"{type(exc).__name__}: {exc}"
        print(f"[send_sms_reminder] {detail}")
        traceback.print_exc()
        return False, detail


def send_voice_reminder(
    phone: str,
    policy_holder: str,
    renewal_date_str: str,
    premium: str,
    language: str = "es",
) -> tuple[bool, str]:
    """Place an outbound voice call that reads the renewal reminder aloud."""
    config = resolve_config(load_config())
    if not is_voice_configured(config):
        return False, "Voice not configured"

    phone = clean_phone_e164(phone.strip())
    if not phone or phone == "+":
        return False, "Phone number empty"

    from_number = get_twilio_voice_from(config)
    _, body = build_reminder_message(policy_holder, renewal_date_str, premium, language)
    say_lang = "es-MX" if language == "es" else "en-US"
    safe_text = xml.sax.saxutils.escape(body)
    twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Say language="{say_lang}">{safe_text}</Say></Response>'

    try:
        from twilio.rest import Client

        client = Client(config["TWILIO_ACCOUNT_SID"], config["TWILIO_AUTH_TOKEN"])
        call = client.calls.create(twiml=twiml, to=phone, from_=from_number)
        detail = f"SID={call.sid} status={call.status} to={phone} from={from_number}"
        print(f"[send_voice_reminder] {detail}")
        return True, f"{call.status}|{call.sid}"
    except Exception as exc:
        try:
            from twilio.base.exceptions import TwilioRestException

            if isinstance(exc, TwilioRestException):
                detail = f"{exc.code} — {exc.msg}"
            else:
                detail = f"{type(exc).__name__}: {exc}"
        except ImportError:
            detail = f"{type(exc).__name__}: {exc}"
        print(f"[send_voice_reminder] {detail}")
        traceback.print_exc()
        return False, detail


def render_env_diagnostics(keys: tuple[str, ...] | None = None) -> None:
    """Show where each config value was loaded from (no secret values exposed)."""
    keys = keys or ("GEMINI_API_KEY", "TWILIO_AUTH_TOKEN", "TWILIO_ACCOUNT_SID")
    file_values = parse_env_file_with_dotenv(ENV_PATH) if ENV_PATH.exists() else {}
    config = load_config()
    env_stats = get_env_file_stats()

    if env_stats["exists"]:
        st.caption(
            t("env_disk_stats").format(
                size=env_stats["size_bytes"],
                modified=env_stats["modified"],
            )
        )
        if env_stats.get("encoding"):
            st.caption(t("env_encoding").format(encoding=env_stats["encoding"]))
        st.caption(t("env_os_ignored"))
    else:
        st.warning(f"{t('env_file_label')}: `{ENV_PATH}` — file not found on disk")

    for key in keys:
        value = config.get(key, "")
        source = get_config_source(key, file_values, value)
        if source == "file":
            source_label = t("env_source_file")
        elif source == "secrets":
            source_label = t("env_source_secrets")
        else:
            source_label = t("env_source_missing")
        status = env_status_label(value)
        char_note = t("env_value_chars").format(count=len(value)) if value else ""
        detail = f" — {char_note}" if char_note else ""
        st.caption(f"`{key}`: {status} ({source_label}){detail}")

    for key in keys:
        note = get_stale_env_note(key, file_values)
        if note:
            st.warning(note)


def render_twilio_token_input() -> None:
    """Single global Twilio token field (sidebar only — must not duplicate keys)."""
    st.markdown("---")
    st.text_input(
        t("twilio_token_manual"),
        type="password",
        key="manual_twilio_auth_token",
        help=t("twilio_token_manual_help"),
    )
    manual_token = get_manual_twilio_token()
    if manual_token:
        st.caption(t("twilio_token_manual_active").format(count=len(manual_token)))
    elif not is_twilio_credentials_ok(load_config()):
        st.caption(t("twilio_token_hint"))


def render_when_sent_help() -> None:
    st.info(t("when_sent_desc"))


def render_messaging_debug_config(config: dict[str, str]) -> None:
    """SMTP/Twilio setup panels — sidebar debug only, not shown in main reminder UI."""
    smtp_fields = {
        "SMTP_HOST": config.get("SMTP_HOST", ""),
        "SMTP_USER": config.get("SMTP_USER", ""),
        "SMTP_PASSWORD": config.get("SMTP_PASSWORD", ""),
        "SMTP_FROM": config.get("SMTP_FROM", ""),
    }

    def status_line(name: str, value: str) -> str:
        status = env_status_label(value)
        if status == "set":
            label = t("env_var_set")
        elif status == "placeholder":
            label = t("env_var_placeholder")
        else:
            label = t("env_var_missing")
        char_note = t("env_value_chars").format(count=len(value)) if value else ""
        suffix = f" ({char_note})" if char_note else ""
        return f"- `{name}`: **{label}**{suffix}"

    with st.expander(t("smtp_config_title"), expanded=not is_smtp_configured(config)):
        if is_smtp_configured(config):
            st.success(t("smtp_config_ok"))
            if smtp_use_ssl(config):
                st.caption(t("smtp_config_ssl_note"))
        else:
            st.info(t("smtp_config_missing"))
            st.markdown("\n".join(status_line(k, v) for k, v in smtp_fields.items()))
            st.code(
                "SMTP_HOST=smtp.gmail.com\n"
                "SMTP_PORT=465\n"
                "SMTP_SSL=true\n"
                "SMTP_USER=your_email@gmail.com\n"
                "SMTP_PASSWORD=your_app_password\n"
                "SMTP_FROM=your_email@gmail.com",
                language="text",
            )

    voice_from = get_twilio_voice_from(resolve_config(config))
    with st.expander(t("voice_config_title"), expanded=not is_voice_configured(resolve_config(config))):
        resolved = resolve_config(config)
        if is_voice_configured(resolved):
            st.success(t("voice_config_ok"))
            st.caption(f"`TWILIO_VOICE_FROM`: `{voice_from}`")
        else:
            st.info(t("voice_config_missing"))
            st.code(
                "TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
                "TWILIO_AUTH_TOKEN=your_real_auth_token_here\n"
                "TWILIO_VOICE_FROM=+15551234567",
                language="text",
            )
            st.caption(t("voice_config_trial_note"))
            st.caption(
                "Use the same Twilio number shown under Phone Numbers if it has **Voice** enabled "
                "(no A2P 10DLC badge)."
            )

    with st.expander(t("sms_config_title"), expanded=not is_sms_configured(resolve_config(config))):
        resolved = resolve_config(config)
        if is_sms_configured(resolved):
            st.success(t("sms_config_ok"))
        else:
            st.info(t("sms_config_missing"))
            st.info(t("messaging_config_sidebar_hint"))
            st.code(
                "TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
                "TWILIO_AUTH_TOKEN=your_real_auth_token_here\n"
                "TWILIO_SMS_FROM=+15551234567",
                language="text",
            )
            st.caption(t("sms_config_trial_note"))
            st.caption(
                "Get a Twilio phone number: Console → Phone Numbers → Manage → Buy a number "
                "(or use your trial number)."
            )

    with st.expander(t("whatsapp_config_title"), expanded=not is_twilio_configured(resolve_config(config))):
        manual_token = get_manual_twilio_token()
        resolved = resolve_config(config)
        if is_twilio_configured(resolved):
            if manual_token:
                st.success(t("whatsapp_config_ok_session").format(count=len(manual_token)))
            else:
                st.success(t("whatsapp_config_ok"))
        else:
            st.info(t("whatsapp_config_missing"))
            st.info(t("messaging_config_sidebar_hint"))
            st.info(t("twilio_token_sidebar_hint"))
            st.code(
                "TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
                "TWILIO_AUTH_TOKEN=your_real_auth_token_here\n"
                "TWILIO_WHATSAPP_FROM=whatsapp:+14155238886",
                language="text",
            )
            st.caption(
                "Twilio sandbox: join your sandbox from the Twilio console, "
                "then use the sandbox number as TWILIO_WHATSAPP_FROM."
            )
        st.caption(t("whatsapp_production_note"))


def parse_stored_analysis(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if row.get("analysis_json"):
        try:
            analysis = json.loads(row["analysis_json"])
        except json.JSONDecodeError:
            analysis = {}
    else:
        analysis = {
            "policy_holder": row.get("policy_holder", ""),
            "insurer": row.get("insurer", ""),
            "renewal_date": row.get("renewal_date", ""),
            "premium_amount": row.get("premium", ""),
            "payment_frequency": row.get("payment_frequency", "annual"),
            "coverage_details": t("not_found"),
            "smart_questions": [],
        }

    if row.get("context_json"):
        try:
            context = json.loads(row["context_json"])
        except json.JSONDecodeError:
            context = {}
    else:
        renewal = analysis.get("renewal_date") or row.get("renewal_date")
        risk_level, risk_msg_key = compute_risk_level(
            renewal if renewal and renewal != "null" else None,
            get_reference_date(),
        )
        context = {
            "risk_level": risk_level,
            "risk_msg_key": risk_msg_key,
            "analysis_language": row.get("language", "en"),
        }

    context["reminders_active"] = bool(row.get("reminders_active", 1))
    context["payment_confirmed_at"] = row.get("payment_confirmed_at")
    context["payment_frequency"] = row.get("payment_frequency") or analysis.get("payment_frequency", "annual")
    context["reminder_start_days"] = row.get("reminder_start_days") or analysis.get("suggested_reminder_start_days", 30)
    context["reminder_base_time"] = row.get("reminder_base_time") or analysis.get("suggested_reminder_time", "09:00")
    context["email_primary"] = (row.get("email_primary") or "").strip()
    context["email_secondary"] = (row.get("email_secondary") or "").strip()
    settings_blob = load_reminder_settings_blob(row, context)
    wa_channel = settings_blob.get("whatsapp") if isinstance(settings_blob.get("whatsapp"), dict) else {}
    context["whatsapp_phone"] = (settings_blob.get("whatsapp_phone") or wa_channel.get("primary") or "").strip()
    context["whatsapp_secondary"] = (settings_blob.get("whatsapp_secondary") or wa_channel.get("secondary") or "").strip()
    context["insurance_id"] = insurance_id_from_row(row)
    context["upload_version"] = int(row.get("upload_version") or 1)
    context["analysis_id"] = row.get("id")
    context["payment_confirmations"] = load_payment_confirmations(row)
    return analysis, context


def format_questions_html(questions: list[str]) -> str:
    if not questions:
        return t("not_found")
    return "".join(f'<div class="question-item">{i + 1}. {q}</div>' for i, q in enumerate(questions))


def format_coverage_html(coverage: str) -> str:
    return coverage.replace("\n", "<br>")


# ---------------------------------------------------------------------------
# UI — analysis & reminders
# ---------------------------------------------------------------------------
def render_reminder_section(
    analysis: dict[str, Any],
    context: dict[str, Any],
    *,
    key_prefix: str = "",
) -> None:
    """Render reminder configuration, plan preview, and payment confirmation."""
    reminders_active = context.get("reminders_active", True)
    analysis_id = context.get("analysis_id")
    row_data: dict[str, Any] | None = None
    if analysis_id:
        row_data = get_analysis_by_id(int(analysis_id))
    sync_reminder_fields_from_policy(key_prefix, row_data, context, analysis)

    if context.get("payment_confirmed_at"):
        st.markdown(
            f'<div class="status-green"><strong>{t("reminders_stopped")}</strong></div>',
            unsafe_allow_html=True,
        )
    elif reminders_active:
        st.markdown(
            f'<div class="status-amber"><strong>{t("reminders_active")}</strong></div>',
            unsafe_allow_html=True,
        )

    ai_start = analysis.get("suggested_reminder_start_days", 30)
    ai_time = analysis.get("suggested_reminder_time", "09:00")
    ai_rationale = sanitize_reminder_rationale(
        analysis.get("reminder_rationale", ""),
        analysis.get("premium_amount", ""),
    )

    if ai_rationale:
        render_card(
            t("section_ai_reminder"),
            f"<strong>{t('reminder_days')}:</strong> {ai_start}<br>"
            f"<strong>{t('reminder_time')}:</strong> {ai_time}<br>{ai_rationale}",
        )

    render_schedule_timing_inputs(key_prefix, analysis_id=analysis_id, context=context)

    email_cfg = read_channel_form_values(key_prefix, "email", analysis=analysis, context=context)
    reminder_days = int(email_cfg["reminder_start_days"])
    reminder_time = email_cfg["reminder_base_time"]
    frequent_days = int(email_cfg["frequent_start_days"])
    daily_frequency = int(email_cfg["daily_frequency"])

    rd, due, premium = get_reminder_due_context(analysis, context)
    holder = analysis.get("policy_holder", "")
    lang = st.session_state.language

    if due and reminders_active:
        try:
            ref = get_reference_date()
            schedule = build_reminder_schedule(due, int(reminder_days), reminder_time, frequent_days, daily_frequency)
            st.markdown(f'<div class="section-heading">{t("reminder_plan_title")}</div>', unsafe_allow_html=True)
            st.caption(t("reminder_plan_desc"))
            plan_col, _plan_spacer = st.columns([1, 1])
            with plan_col:
                with st.expander(t("reminder_plan_view"), expanded=False):
                    plan_rows = []
                    for r in schedule[:60]:
                        row = {
                            t("plan_col_date"): r["date"],
                            t("plan_col_time"): r["time"],
                            t("plan_col_tier"): tier_label(r["tier"], r["count"]),
                        }
                        if r["date"] == ref.isoformat():
                            row[t("plan_col_date")] = f"{r['date']} *"
                            row[t("plan_col_tier")] = f"{tier_label(r['tier'], r['count'])} ({t('sim_due_now')})"
                        plan_rows.append(row)
                    st.markdown('<div class="schedule-table-marker"></div>', unsafe_allow_html=True)
                    st.dataframe(plan_rows, width="stretch", hide_index=True, height=190)
                    st.caption(t("reminder_plan_scroll_hint"))
                    if ref != date.today():
                        due_on_ref = reminders_on_date(due, ref, int(reminder_days), reminder_time, frequent_days, daily_frequency)
                        if due_on_ref:
                            st.info(f"{t('sim_due_now')}: {len(due_on_ref)} — {tier_label(due_on_ref[0]['tier'], due_on_ref[0]['count'])}")
                    if len(schedule) > 60:
                        st.caption(f"+ {len(schedule) - 60} more reminders")

                    ics_bytes = generate_ics_schedule(schedule, analysis.get("policy_holder", ""), st.session_state.language)
                    st.download_button(
                        label=t("download_ics"),
                        data=ics_bytes,
                        file_name="renewal_reminders.ics",
                        mime="text/calendar",
                        key=f"{key_prefix}ics_download",
                    )
        except ValueError:
            st.warning(t("err_no_renewal"))

    config = load_config()
    render_when_sent_help()
    config = resolve_config(config)

    render_compact_channel_recipients(
        "email", key_prefix, analysis_id=analysis_id, context=context,
    )
    render_compact_channel_recipients(
        "whatsapp", key_prefix, analysis_id=analysis_id, context=context,
    )

    st.markdown(f'<div class="section-heading">{t("send_now_title")}</div>', unsafe_allow_html=True)
    st.caption(t("send_now_desc"))

    phone = st.session_state.get(f"{key_prefix}whatsapp_phone", "")

    send_left, send_right = st.columns(2)
    with send_left:
        st.markdown('<div class="send-now-actions-marker"></div>', unsafe_allow_html=True)
        if is_smtp_configured(config):
            if st.button(
                t("send_email_now"),
                key=f"{key_prefix}send_email_btn",
                disabled=not reminders_active,
                width="stretch",
            ):
                emails = [
                    st.session_state.get(f"{key_prefix}email_primary", ""),
                    st.session_state.get(f"{key_prefix}email_secondary", ""),
                ]
                if not emails[0].strip():
                    st.warning(t("email_no_address"))
                elif rd and rd != "null":
                    if send_email_reminder(emails, holder, rd, premium, lang):
                        st.success(t("email_sent"))
                    else:
                        st.error(t("email_failed"))
                else:
                    st.warning(t("err_no_renewal"))
        else:
            st.info(t("email_not_configured"))

        if is_sms_configured(config):
            if st.button(
                t("send_sms_now"),
                key=f"{key_prefix}send_sms_btn",
                disabled=not reminders_active,
                width="stretch",
            ):
                if rd and rd != "null" and phone.strip():
                    ok, detail = send_sms_reminder(phone, holder, rd, premium, lang)
                    if ok:
                        status, sid = (detail.split("|", 1) + [""])[:2]
                        st.success(t("sms_sent"))
                        st.caption(t("sms_sent_detail").format(status=status, sid=sid))
                        st.caption(t("sms_sent_note"))
                        st.markdown(
                            "[Twilio Message Logs](https://console.twilio.com/us1/monitor/logs/messages)"
                        )
                    else:
                        st.error(t("sms_failed"))
                        st.code(t("sms_failed_detail").format(detail=detail), language="text")
                        if "30044" in detail:
                            st.info(t("twilio_trial_length_hint"))
                        if "21219" in detail:
                            sent = clean_phone_e164(phone.strip())
                            st.info(t("twilio_verified_mismatch").format(sent=sent))
                elif not phone.strip():
                    st.warning(t("sms_no_phone"))
                else:
                    st.warning(t("err_no_renewal"))
        else:
            st.info(t("sms_not_configured"))

    with send_right:
        st.markdown('<div class="send-now-actions-marker"></div>', unsafe_allow_html=True)
        if is_twilio_configured(config):
            if st.button(
                t("send_whatsapp_now"),
                key=f"{key_prefix}send_whatsapp_btn",
                disabled=not reminders_active,
                width="stretch",
            ):
                if rd and rd != "null" and phone.strip():
                    ok, detail = send_whatsapp_reminder(phone, holder, rd, premium, lang)
                    if ok:
                        status, sid = (detail.split("|", 1) + [""])[:2]
                        st.success(t("whatsapp_sent"))
                        st.caption(t("whatsapp_sent_detail").format(status=status, sid=sid))
                        st.caption(t("whatsapp_sent_note"))
                        st.markdown(
                            "[Twilio Message Logs](https://console.twilio.com/us1/monitor/logs/messages)"
                        )
                    else:
                        st.error(t("whatsapp_failed"))
                        st.code(t("whatsapp_failed_detail").format(detail=detail), language="text")
                        if "30044" in detail:
                            st.info(t("twilio_trial_length_hint"))
                        if "63015" in detail:
                            st.info(t("twilio_sandbox_join"))
                            st.code(f"Sent to: whatsapp:{clean_phone_e164(phone.strip())}", language="text")
                elif not phone.strip():
                    st.warning(t("sms_no_phone"))
                else:
                    st.warning(t("err_no_renewal"))
        elif not (rd and rd != "null" and phone.strip() and reminders_active):
            st.info(t("whatsapp_available_soon"))

        if rd and rd != "null" and phone.strip() and reminders_active:
            _, wa_body = build_reminder_message(holder, rd, premium, lang)
            st.markdown('<div class="whatsapp-demo-link-marker"></div>', unsafe_allow_html=True)
            st.link_button(
                t("whatsapp_demo_open"),
                build_whatsapp_deeplink(phone, wa_body),
                width="stretch",
                help=t("whatsapp_demo_help"),
            )
            st.caption(t("whatsapp_demo_help"))


def render_analysis_results(
    analysis: dict[str, Any],
    context: dict[str, Any],
    *,
    show_reminders: bool = True,
    key_prefix: str = "",
) -> None:
    risk_level = context.get("risk_level", "ok")
    risk_msg_key = context.get("risk_msg_key", "risk_ok")
    renewal = analysis.get("renewal_date")
    if renewal and renewal != "null":
        risk_level, risk_msg_key = compute_risk_level(renewal, get_reference_date())
        context["risk_level"] = risk_level
        context["risk_msg_key"] = risk_msg_key
    if is_current_term_paid(context, analysis):
        risk_level = "ok"
        risk_msg_key = "risk_ok"
        context["risk_level"] = risk_level
        context["risk_msg_key"] = risk_msg_key
    analysis_lang = context.get("analysis_language", st.session_state.language)
    upload_version = int(context.get("upload_version") or context.get("pending_upload_version") or 1)
    insurance_id = context.get("insurance_id") or get_insurance_id(analysis)
    render_version_badge(upload_version, insurance_id)

    if analysis_lang != st.session_state.language:
        lang_name = t("lang_name_en") if analysis_lang == "en" else t("lang_name_es")
        st.info(t("analysis_language_note").format(lang=lang_name))

    policy_holder = analysis.get("policy_holder") or t("not_found")
    st.markdown(
        f'<div class="policy-holder-line">{t("section_policy_holder")}: '
        f"<strong>{policy_holder}</strong></div>",
        unsafe_allow_html=True,
    )

    render_policy_detail_grid(analysis, context)

    render_card(t("section_coverage"), format_coverage_html(analysis.get("coverage_details", t("not_found"))))

    st.markdown(f'<div class="section-heading">{t("section_questions")}</div>', unsafe_allow_html=True)
    st.caption(t("section_questions_desc"))
    with st.expander(t("section_questions_view"), expanded=False):
        st.markdown(
            f'<div class="card-body">{format_questions_html(analysis.get("smart_questions", []))}</div>',
            unsafe_allow_html=True,
        )

    render_payment_subrecords(analysis, context, key_prefix=key_prefix)
    if not analysis.get("renewal_date") or analysis.get("renewal_date") == "null":
        st.warning(t("err_no_renewal"))

    if show_reminders:
        st.markdown(f'<div class="section-heading">{t("reminder_title")}</div>', unsafe_allow_html=True)
        render_reminder_section(analysis, context, key_prefix=key_prefix)


def render_persisted_analysis() -> None:
    analysis = st.session_state.get("last_analysis")
    if not analysis:
        return
    context = st.session_state.get("analysis_context", {})
    if st.session_state.get("saved_analysis_id"):
        context["analysis_id"] = st.session_state.saved_analysis_id
    holder = analysis.get("policy_holder", t("not_found"))
    version = int(context.get("upload_version") or context.get("pending_upload_version") or 1)
    policy_num = analysis.get("policy_number") or context.get("insurance_id") or get_insurance_id(analysis) or t("not_found")
    show_reminders = not st.session_state.get("duplicate_pending", False)
    analysis_id = st.session_state.get("saved_analysis_id")
    key_prefix = reminder_key_prefix(analysis_id)
    expander_label = format_policy_header_label(
        policy_num=policy_num,
        holder=holder,
        insurance_type=analysis.get("insurance_type") or t("not_found"),
        renewal_date=analysis.get("renewal_date"),
        version=version,
    )
    with st.expander(expander_label, expanded=True):
        render_analysis_results(
            analysis, context, show_reminders=show_reminders, key_prefix=key_prefix
        )


def store_analysis_session(analysis: dict[str, Any], risk_level: str, risk_msg_key: str) -> None:
    freq = normalize_payment_frequency(analysis.get("payment_frequency"))
    insurance_id = get_insurance_id(analysis)
    existing = find_existing_uploads(analysis)
    pending_version = next_upload_version(analysis) if existing else 1
    st.session_state.last_analysis = analysis
    st.session_state.analysis_context = {
        "risk_level": risk_level,
        "risk_msg_key": risk_msg_key,
        "analysis_language": st.session_state.language,
        "payment_frequency": freq,
        "reminder_start_days": analysis.get("suggested_reminder_start_days", 30),
        "reminder_base_time": analysis.get("suggested_reminder_time", "09:00"),
        "reminders_active": True,
        "insurance_id": insurance_id,
        "pending_upload_version": pending_version,
    }
    st.session_state.pop("show_replace_manual_hint", None)


def process_new_analysis(analysis: dict[str, Any], risk_level: str, risk_msg_key: str) -> bool:
    """Run immediately after parsing. Returns True when saved to the database."""
    store_analysis_session(analysis, risk_level, risk_msg_key)
    existing = find_existing_uploads(analysis)
    context = st.session_state.analysis_context
    insurance_id = get_insurance_id(analysis)

    if existing:
        st.session_state.duplicate_pending = True
        st.session_state.duplicate_existing_count = len(existing)
        st.session_state.analysis_db_saved = False
        context["insurance_id"] = insurance_id
        context["pending_upload_version"] = next_upload_version(analysis)
        st.session_state.analysis_context = context
        return False

    st.session_state.duplicate_pending = False
    st.session_state.pop("duplicate_existing_count", None)
    persist_analysis_to_db(analysis, context, 1)
    return True


@st.dialog(" ")
def confirm_delete_dialog() -> None:
    st.markdown(f"**{t('delete_confirm_title')}**")
    st.warning(t("delete_irreversible"))
    col_yes, col_no = st.columns(2)
    pending_id = st.session_state.get("delete_pending_id")
    with col_yes:
        if st.button(t("delete_yes"), type="primary", width="stretch"):
            if pending_id is not None:
                delete_from_database(int(pending_id))
                if st.session_state.get("saved_analysis_id") == pending_id:
                    clear_current_analysis()
                st.session_state.pop("delete_pending_id", None)
                st.session_state.delete_flash = True
                st.rerun()
    with col_no:
        if st.button(t("delete_no"), width="stretch"):
            st.session_state.pop("delete_pending_id", None)
            st.rerun()


def clear_current_analysis() -> None:
    for key in (
        "last_analysis",
        "analysis_context",
        "saved_analysis_id",
        "analysis_db_saved",
        "duplicate_pending",
        "duplicate_existing_count",
        "show_replace_manual_hint",
    ):
        st.session_state.pop(key, None)


def render_history_section() -> None:
    st.markdown("---")
    st.markdown(f"**{t('history_title')}**")
    st.caption(t("history_load_hint"))
    if st.session_state.pop("delete_flash", False):
        st.success(t("delete_success"))
    history = get_history()
    if not history:
        st.caption(t("history_empty"))
        return

    reference = get_reference_date()
    for row in history:
        row_id = int(row["id"])
        analysis, context = parse_stored_analysis(row)
        label = format_history_label_compact(row)
        renewal_color, renewal_text = get_history_renewal_status(analysis, reference)
        payment_color, payment_text = get_history_payment_status(analysis, context, reference)

        col_main, _spacer, col_delete = st.columns([7, 2.8, 0.5])
        with col_main:
            st.markdown('<div class="history-policy-item-marker"></div>', unsafe_allow_html=True)
            st.markdown('<div class="history-row-marker"></div>', unsafe_allow_html=True)
            with st.expander(label, expanded=False):
                render_analysis_results(
                    analysis, context, show_reminders=True, key_prefix=reminder_key_prefix(row_id, scope="history")
                )
            st.markdown('<div class="history-status-section-marker"></div>', unsafe_allow_html=True)
            status_col1, status_col2 = st.columns(2, gap="medium")
            with status_col1:
                st.markdown(f'<div class="history-table-head">{t("history_col_renewal")}</div>', unsafe_allow_html=True)
                render_history_status_chip(renewal_color, renewal_text)
            with status_col2:
                st.markdown(f'<div class="history-table-head">{t("history_col_payment")}</div>', unsafe_allow_html=True)
                render_history_status_chip(payment_color, payment_text)
        with col_delete:
            st.markdown("<div style='height: 0.15rem'></div>", unsafe_allow_html=True)
            if st.button("", icon=":material/delete:", key=f"delete_{row_id}", help=t("delete_btn_help")):
                st.session_state.delete_pending_id = row_id
                confirm_delete_dialog()


def restart_app() -> None:
    language = st.session_state.get("language", "en")
    st.session_state.clear()
    st.session_state.language = language
    st.session_state.restart_flash = True


def main() -> None:
    st.set_page_config(page_title="Insurance Renewal Guardian v2", layout="wide", initial_sidebar_state="expanded")
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "upload_widget_key" not in st.session_state:
        st.session_state.upload_widget_key = 0
    if st.session_state.get("policy_management_view") not in {"new", "active"}:
        st.session_state.policy_management_view = "new"

    if "renewal_alert_days" not in st.session_state:
        st.session_state.renewal_alert_days = 30
    if "renewal_red_flag_days" not in st.session_state:
        st.session_state.renewal_red_flag_days = 15

    init_db()
    backfill_insurance_ids()
    global_defaults = load_global_contacts()
    if "global_email_primary" not in st.session_state:
        st.session_state.global_email_primary = global_defaults["email_primary"]
    if "global_whatsapp_primary" not in st.session_state:
        st.session_state.global_whatsapp_primary = global_defaults["whatsapp_primary"]

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    config = load_config()
    api_key = get_api_key()
    model_on_disk = config["GEMINI_MODEL"]

    with st.sidebar:
        st.radio(
            t("language_label"), options=["en", "es"],
            format_func=lambda x: "English" if x == "en" else "Español",
            horizontal=True, key="language",
        )
        st.markdown("---")
        render_sidebar_renewal_alerts()
        st.markdown("---")
        render_sidebar_global_contacts()
        st.markdown("---")
        st.markdown(f'<div class="sidebar-title">{t("app_name")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-tagline">{t("tagline")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-version">{t("app_version")}</div>', unsafe_allow_html=True)
        with st.expander(t("sidebar_config_title"), expanded=False):
            st.caption(f'{t("env_file_label")}: `{ENV_PATH}`')
            st.caption(t("env_file_hint").format(name=ENV_FILE_NAME))
            render_env_diagnostics((
                "GEMINI_API_KEY",
                "TWILIO_ACCOUNT_SID",
                "TWILIO_AUTH_TOKEN",
                "TWILIO_WHATSAPP_FROM",
                "TWILIO_SMS_FROM",
                "TWILIO_VOICE_FROM",
            ))
            st.caption(f'{t("env_model_on_disk")}: `{model_on_disk}`')
            if api_key:
                st.caption(t("env_key_ok"))
            else:
                st.warning(t("env_key_missing"))
                st.caption(t("env_disk_hint"))
            render_twilio_token_input()
            st.markdown("---")
            render_messaging_debug_config(resolve_config(load_config()))
        st.markdown(f"**{t('sidebar_steps_title')}**")
        for i, step in enumerate(["step_1", "step_2", "step_3", "step_4"], start=1):
            st.markdown(f'<div class="sidebar-step">{i}. {t(step)}</div>', unsafe_allow_html=True)
        st.markdown("---")
        if st.button(t("restart_btn"), width="stretch", key="restart_btn"):
            restart_app()
            st.rerun()

    if st.session_state.pop("restart_flash", False):
        st.success(t("restart_done"))

    header_left, header_right = st.columns([2.2, 1.3])
    with header_left:
        st.markdown(f'<div class="product-title">{t("product_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="main-header">{t("app_name")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="main-subheader">{t("tagline")} · {t("app_version")}</div>', unsafe_allow_html=True)
    with header_right:
        render_header_today_and_simulation(resolve_config(config))

    st.markdown('<div class="policy-mgmt-radio-marker"></div>', unsafe_allow_html=True)
    st.radio(
        t("policy_mgmt_title"),
        options=["new", "active"],
        format_func=lambda x: {
            "new": t("policy_mgmt_new"),
            "active": t("policy_mgmt_active"),
        }[x],
        horizontal=True,
        key="policy_management_view",
        label_visibility="collapsed",
    )

    if st.session_state.get("policy_management_view", "new") == "new":
        upload_col, _upload_spacer = st.columns([1, 2])
        with upload_col:
            st.markdown('<div class="upload-compact-marker"></div>', unsafe_allow_html=True)
            current_pdf = st.file_uploader(
                t("upload_current"),
                type=["pdf"],
                help=t("upload_current_help"),
                key=f"current_pdf_{st.session_state.upload_widget_key}",
                accept_multiple_files=False,
            )
            st.markdown(
                f'<div class="sim-field-label upload-file-hint">{t("upload_file_hint")}</div>',
                unsafe_allow_html=True,
            )

            if st.button(t("analyze_btn"), type="primary", disabled=not current_pdf):
                st.session_state.pop("pending_analysis", None)
                st.session_state.pop("pending_missing_fields", None)
                current_text, err, detail = extract_text(current_pdf)
                if err:
                    st.session_state.upload_parse_error = {
                        "message": t(err),
                        "detail": detail or "",
                    }
                    st.rerun()
                with st.spinner("..."):
                    analysis, api_err, api_debug = analyze_with_gemini(current_text, st.session_state.language)
                if api_err:
                    st.session_state.upload_parse_error = {
                        "message": t(api_err),
                        "detail": api_debug or "",
                    }
                    st.rerun()
                analysis = normalize_policy_holder(analysis or {})
                missing = get_missing_required_fields(analysis)
                if missing:
                    st.session_state.pending_analysis = analysis
                    st.session_state.pending_missing_fields = missing
                    st.rerun()
                renewal_date = analysis.get("renewal_date") or ""
                risk_level, risk_msg_key = compute_risk_level(
                    renewal_date if renewal_date != "null" else None,
                    get_reference_date(),
                )
                saved = process_new_analysis(analysis, risk_level, risk_msg_key)
                if saved:
                    st.success(t("analysis_saved"))

    render_simulation_results_section()

    if st.session_state.get("upload_parse_error"):
        upload_parse_error_dialog()

    if st.session_state.get("pending_missing_fields"):
        render_missing_fields_form()

    if st.session_state.pop("replace_manual_flash", False):
        st.warning(t("duplicate_delete_manual"))

    if st.session_state.get("duplicate_pending"):
        render_duplicate_alert_panel()

    render_persisted_analysis()

    render_history_section()


if __name__ == "__main__":
    main()
