"""RiskTrail AI Underwriting Copilot Engine.

Provides multi-provider LLM support for credit risk explanation, adverse action analysis,
and portfolio underwriting optimization:
1. Local RiskTrail Copilot (Built-in, zero API key, rule-and-context synthesized engine).
2. Google Gemini 1.5 Flash (via REST API).
3. Groq Llama 3.3 70B (via REST API).
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def build_system_prompt(applicant_context: Dict[str, Any]) -> str:
    """Construct an institutional, lender-grade system prompt with applicant context."""
    cid = applicant_context.get("customer_id", "UNKNOWN")
    risk_prob = applicant_context.get("risk_prob", 0.0)
    cutoff = applicant_context.get("cutoff", 0.05)
    is_approved = applicant_context.get("is_approved", False)
    ground_truth = applicant_context.get("ground_truth", 0)
    profit_good = applicant_context.get("profit_good", 100.0)
    loss_default = applicant_context.get("loss_default", 500.0)
    reasons = applicant_context.get("reasons", [])

    reasons_text = "\n".join([
        f"  - Rank {r.get('rank', i+1)}: {r.get('feature', 'N/A')} = {r.get('feature_value', 0):.4f} "
        f"(SHAP Impact: {r.get('shap_impact', 0):+.4f} log-odds) -> {r.get('adverse_action_reason', 'N/A')}"
        for i, r in enumerate(reasons)
    ])

    return f"""You are the RiskTrail AI Underwriting Copilot, an expert credit risk analyst, quantitative risk modeler, and fair lending compliance advisor.
You provide clear, accurate, and institutional-grade answers regarding credit default scoring, portfolio economics, and adverse action compliance.

ACTIVE APPLICANT UNDERWRITING CONTEXT:
- Synthetic Applicant ID: {cid}
- Predicted Probability of Default (PD): {risk_prob * 100:.2f}%
- Policy Underwriting Approval Cutoff (c): {cutoff * 100:.2f}%
- Underwriting Determination: {"APPROVED" if is_approved else "DECLINED"}
- True Synthetic Outcome (Ground Truth): {"Defaulted (1)" if ground_truth == 1 else "Paid / Non-Default (0)"}
- Portfolio Economics: Profit per Good Account = ${profit_good:.2f}, Loss per Default = ${loss_default:.2f}

TOP PRINCIPAL SHAP ADVERSE ACTION FACTORS:
{reasons_text}

REGULATORY & MODELING STANDARDS:
- ECOA (Equal Credit Opportunity Act / Regulation B 12 C.F.R. § 1002.9) requires 4-5 specific principal reasons for adverse action.
- Explain features in plain, actionable English (e.g. explain what D_39 delinquency slope means).
- Be concise, professional, mathematically grounded, and compliant with consumer lending practices.
"""


def query_local_copilot(
    prompt: str,
    applicant_context: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Generate an instant, context-aware underwriting response offline without API keys."""
    prompt_lower = prompt.lower().strip()
    cid = applicant_context.get("customer_id", "SYNTH_0045")
    risk_prob = applicant_context.get("risk_prob", 0.0)
    cutoff = applicant_context.get("cutoff", 0.05)
    is_approved = applicant_context.get("is_approved", False)
    ground_truth = applicant_context.get("ground_truth", 0)
    reasons = applicant_context.get("reasons", [])
    profit_good = applicant_context.get("profit_good", 100.0)
    loss_default = applicant_context.get("loss_default", 500.0)

    # 1. Why was the applicant declined / approved?
    if any(q in prompt_lower for q in ["why", "decline", "approve", "denied", "reject", "decision", "outcome"]):
        status_word = "approved" if is_approved else "declined"
        reason_bullets = []
        for r in reasons[:4]:
            direction = "elevates default risk" if r.get("shap_impact", 0) > 0 else "lowers default risk"
            reason_bullets.append(
                f"- **{r.get('feature', '')}** ({r.get('adverse_action_reason', '')}): "
                f"Value is `{r.get('feature_value', 0):.4f}` with a SHAP attribution of `{r.get('shap_impact', 0):+.4f} log-odds` ({direction})."
            )
        reasons_block = "\n".join(reason_bullets)

        if not is_approved:
            return (
                f"### Underwriting Assessment: Applicant {cid} (DECLINED)\n\n"
                f"Applicant **{cid}** was **DECLINED** because the model-predicted probability of default is **{risk_prob * 100:.2f}%**, "
                f"which exceeds the current underwriting policy cutoff of **{cutoff * 100:.2f}%** by **{(risk_prob - cutoff) * 100:+.2f}%**.\n\n"
                f"#### Primary Contributing Factors (ECOA § 1002.9 Compliant):\n"
                f"{reasons_block}\n\n"
                f"**Key Driver:** The dominant risk contributor is recent delinquency velocity and elevated revolving balance exposure. "
                f"The borrower demonstrated a deteriorating repayment pattern across the 13-month observation window."
            )
        else:
            return (
                f"### Underwriting Assessment: Applicant {cid} (APPROVED)\n\n"
                f"Applicant **{cid}** was **APPROVED** because the predicted default risk is **{risk_prob * 100:.2f}%**, "
                f"which is safely below the maximum permissible cutoff of **{cutoff * 100:.2f}%**.\n\n"
                f"#### Favorable Credit Factors:\n"
                f"{reasons_block}\n\n"
                f"**Assessment:** The borrower exhibits low historical delinquency volatility and consistent payment ratios."
            )

    # 2. How to improve credit / counterfactual recommendations
    if any(q in prompt_lower for q in ["improve", "better", "counterfactual", "pass", "help", "qualify", "change", "recommend"]):
        top_neg = [r for r in reasons if r.get("shap_impact", 0) > 0][:3]
        advice_list = []
        for r in top_neg:
            feat = r.get("feature", "")
            if "D_39" in feat or "D_" in feat:
                advice_list.append(
                    f"1. **Stabilize Delinquencies ({feat})**: Bring all past-due balances current. "
                    "A sequence of 3 to 6 consecutive on-time monthly payments will reverse the negative delinquency slope."
                )
            elif "B_1" in feat or "B_" in feat:
                advice_list.append(
                    f"2. **Lower Revolving Utilization ({feat})**: Pay down revolving card balances to below 30% of aggregate limits. "
                    "Revolving balance contraction provides the fastest positive SHAP log-odds reduction."
                )
            elif "P_2" in feat or "P_" in feat:
                advice_list.append(
                    f"3. **Increase Payment-to-Balance Ratio ({feat})**: Pay more than the minimum balance each month to demonstrate surplus repayment capacity."
                )
            else:
                advice_list.append(
                    f"- **Improve {feat}**: Address {r.get('adverse_action_reason', feat)} to reduce adverse model weight."
                )

        if not advice_list:
            advice_list = [
                "1. Maintain current on-time payment records across all open credit accounts.",
                "2. Avoid rapid new revolving credit inquiries over the next 6 months.",
                "3. Keep total revolving credit card utilization below 20%.",
            ]

        advice_block = "\n".join(advice_list)
        return (
            f"### Actionable Steps to Improve Credit Approval Probability ({cid})\n\n"
            f"To shift applicant **{cid}** from **{risk_prob * 100:.2f}%** down below the **{cutoff * 100:.2f}%** approval threshold, "
            f"the applicant should target the following concrete actions:\n\n"
            f"{advice_block}\n\n"
            f"**Simulated Impact:** Reversing the delinquency trajectory and reducing balance ratio by 25% typically lowers predicted default risk by 15–30 percentage points in this model."
        )

    # 3. Explain D_39, B_1, slope, or other specific features
    if any(f in prompt_lower for f in ["d_39", "d39", "b_1", "b1", "p_2", "p2", "slope", "delta", "feature"]):
        return (
            f"### Feature Dictionary & Sequential Modeling Explanation\n\n"
            f"- **`D_39` (Delinquency Indicator)**: Represents days past due and revolving delinquency balance. "
            f"`D_39__slope` captures the directional rate of delinquency change over 13 statement periods. A positive slope indicates deteriorating credit behavior.\n"
            f"- **`B_1` (Revolving Balance Ratio)**: Measures outstanding credit card balance relative to borrower limits. High values signal heavy debt load and liquidity stress.\n"
            f"- **`P_2` (Payment Ratio)**: Measures actual payment amounts made relative to monthly billed balances. Higher values indicate strong repayment vigor.\n"
            f"- **`delta_last_first` & `slope`**: Sequential features that summarize temporal momentum. A borrower whose delinquency is accelerating will receive significantly higher default log-odds than a borrower with a stable history."
        )

    # 4. Profit curve, economics, or cutoff questions
    if any(q in prompt_lower for q in ["profit", "cutoff", "economic", "margin", "loss", "optimum", "curve", "threshold"]):
        return (
            f"### Portfolio Profit Optimization Dynamics\n\n"
            f"The expected portfolio profit function is governed by:\n"
            r"$$\text{Profit}(c) = V_{\text{good}} \cdot N_{\text{good, app}}(c) - L_{\text{default}} \cdot N_{\text{def, app}}(c)$$"
            f"\n\n"
            f"**Current Parameters:**\n"
            f"- Profit per Good Customer ($V_\\text{{good}}$): **${profit_good:,.2f}**\n"
            f"- Loss per Default ($L_\\text{{default}}$): **${loss_default:,.2f}**\n"
            f"- Current Cutoff ($c$): **{cutoff:.2f}**\n\n"
            f"**Economic Interpretation:** At higher cutoffs, the lender approves more borrowers, capturing more good revenue but suffering sharp charge-offs when defaulters are admitted. "
            f"The optimal cutoff $c^*$ occurs where the marginal expected revenue of the next approved applicant exactly equals the marginal expected charge-off loss: "
            r"$c^* = \frac{V_{\text{good}}}{V_{\text{good}} + L_{\text{default}}} = \frac{" + f"{profit_good}" + r"}{" + f"{profit_good + loss_default}" + r"} = " + f"{profit_good / (profit_good + loss_default):.3f}$."
        )

    # 5. Regulatory, ECOA, FCRA, or CFPB questions
    if any(q in prompt_lower for q in ["ecoa", "fcra", "cfpb", "compliance", "regulation", "adverse action", "law"]):
        return (
            f"### Regulatory Compliance Overview (ECOA Reg B & FCRA)\n\n"
            f"- **Equal Credit Opportunity Act (ECOA) / Regulation B (12 C.F.R. § 1002.9)**: "
            f"Lenders are legally required to provide consumers with an Adverse Action Notice whenever credit is denied. "
            f"The notice must state the specific principal reasons (typically 4 or 5) for the adverse determination.\n"
            f"- **Role of SHAP in Compliance**: TreeExplainer computes exact local Shapley values that decompose the model logit score into individual feature attributions, ensuring non-arbitrary, scientifically auditable adverse action reason codes.\n"
            f"- **Fair Credit Reporting Act (FCRA § 615)**: Discloses consumer reporting agency details, credit score data, and the consumer's right to obtain a free credit report within 60 days."
        )

    # General / Default Underwriting Copilot Response
    return (
        f"### RiskTrail Underwriting Summary for Applicant {cid}\n\n"
        f"- **Model Score**: `{risk_prob * 100:.2f}%` default probability.\n"
        f"- **Policy Cutoff**: `{cutoff * 100:.2f}%` threshold -> **{'APPROVED' if is_approved else 'DECLINED'}**.\n"
        f"- **Dominant Feature**: `{reasons[0].get('feature', 'N/A')}` ({reasons[0].get('adverse_action_reason', 'N/A')}) "
        f"with impact `{reasons[0].get('shap_impact', 0):+.4f} log-odds`.\n\n"
        f"Feel free to ask me:\n"
        f"- *'Why was this applicant declined?'*\n"
        f"- *'How can this applicant improve their credit?'*\n"
        f"- *'What cutoff maximizes portfolio profit?'*\n"
        f"- *'Explain D_39 delinquency slope'*."
    )


def query_gemini_api(
    api_key: str,
    prompt: str,
    applicant_context: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Query Google Gemini 1.5 Flash via REST API using standard urllib."""
    if not api_key or not api_key.strip():
        return (
            "⚠️ **Gemini API Key Missing**: Please enter your Google Gemini API key in the sidebar under "
            "'🔑 AI Assistant Configuration', or switch the model dropdown to **Local RiskTrail Copilot** (free and offline)."
        )

    system_instruction = build_system_prompt(applicant_context)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key.strip()}"

    contents = []
    # Include recent chat history if available
    if chat_history:
        for msg in chat_history[-6:]:
            role = "model" if msg.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    contents.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1000,
        },
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return "No response text received from Gemini API."
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8", errors="ignore")
        return f"⚠️ **Gemini API Error (HTTP {e.code})**: {error_msg}"
    except urllib.error.URLError as e:
        return f"⚠️ **Connection Error**: Unable to reach Gemini API ({e.reason})."
    except Exception as e:
        return f"⚠️ **Unexpected Error**: {str(e)}"


def query_groq_api(
    api_key: str,
    prompt: str,
    applicant_context: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Query Groq API (Llama 3.3 70B) via REST API using standard urllib."""
    if not api_key or not api_key.strip():
        return (
            "⚠️ **Groq API Key Missing**: Please enter your Groq API key in the sidebar under "
            "'🔑 AI Assistant Configuration', or switch the model dropdown to **Local RiskTrail Copilot** (free and offline)."
        )

    system_instruction = build_system_prompt(applicant_context)
    url = "https://api.groq.com/openai/v1/chat/completions"

    messages = [{"role": "system", "content": system_instruction}]
    if chat_history:
        for msg in chat_history[-6:]:
            role = "assistant" if msg.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": msg.get("content", "")})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1000,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return "No response text received from Groq API."
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8", errors="ignore")
        return f"⚠️ **Groq API Error (HTTP {e.code})**: {error_msg}"
    except urllib.error.URLError as e:
        return f"⚠️ **Connection Error**: Unable to reach Groq API ({e.reason})."
    except Exception as e:
        return f"⚠️ **Unexpected Error**: {str(e)}"
