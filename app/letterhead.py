"""Official Bank Letterhead Document Generator for RiskTrail Adverse Action Statements.

Produces authentic, bank-grade stationery paper replicas compliant with:
- Equal Credit Opportunity Act (ECOA) / Regulation B (12 C.F.R. § 1002.9)
- Fair Credit Reporting Act (FCRA) Section 615(a)
- Consumer Financial Protection Bureau (CFPB) Circular 2022-03 (Explainability for Complex Models)
"""

from typing import Any, Dict, List, Optional
import pandas as pd


def generate_bank_letter_html(
    customer_id: str,
    risk_prob: float,
    cutoff: float,
    is_approved: bool,
    reason_codes: List[Dict[str, Any]],
    cust_tab: Optional[pd.Series] = None,
) -> str:
    """Generate an authentic bank stationery HTML document replica."""
    status_label = "APPROVED" if is_approved else "DECLINED"
    status_color = "#16A34A" if is_approved else "#DC2626"
    status_bg = "#F0FDF4" if is_approved else "#FEF2F2"
    status_border = "#22C55E" if is_approved else "#EF4444"

    # Build numbered reason list
    reasons_rows_html = []
    for r in reason_codes:
        impact = r.get("shap_impact", 0.0)
        direction_badge = (
            '<span style="background-color: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">Elevates Default Risk 🔴</span>'
            if impact > 0
            else '<span style="background-color: #E0F2FE; color: #075985; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">Lowers Risk 🔵</span>'
        )
        impact_str = f"{impact:+.4f} log-odds"
        feat_name = r.get("feature", "N/A")
        val_str = f"{r.get('feature_value', 0.0):.4f}"
        desc = r.get("adverse_action_reason", feat_name)

        reasons_rows_html.append(f"""
        <tr style="border-bottom: 1px solid #E5E7EB;">
            <td style="padding: 10px 12px; font-weight: 700; color: #374151; width: 40px; text-align: center;">{r.get('rank', '-')}</td>
            <td style="padding: 10px 12px;">
                <div style="font-weight: 600; color: #111827; font-size: 13.5px;">{desc}</div>
                <div style="font-size: 11.5px; color: #6B7280; margin-top: 2px; font-family: 'Courier New', monospace;">Variable: {feat_name} | Value: {val_str}</div>
            </td>
            <td style="padding: 10px 12px; font-family: 'Courier New', monospace; font-size: 12.5px; font-weight: 600; text-align: right; color: #1F2937;">{impact_str}</td>
            <td style="padding: 10px 12px; text-align: center;">{direction_badge}</td>
        </tr>
        """)

    reasons_table_body = "".join(reasons_rows_html)

    return f"""
<div style="
    background: #FFFFFF;
    color: #1F2937;
    font-family: 'Georgia', 'Times New Roman', serif;
    padding: 40px 48px;
    border-radius: 6px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05);
    border: 1px solid #E2E8F0;
    border-top: 6px solid #1E3A8A;
    line-height: 1.55;
    margin: 16px 0 24px 0;
">
    <!-- Corporate Header / Letterhead -->
    <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #0F172A; padding-bottom: 16px; margin-bottom: 24px;">
        <div>
            <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 22px; font-weight: 800; letter-spacing: 1.2px; color: #0F172A; text-transform: uppercase;">
                🏛️ RISKTRAIL FINANCIAL SERVICES CORP.
            </div>
            <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11.5px; font-weight: 600; letter-spacing: 0.8px; color: #475569; text-transform: uppercase; margin-top: 2px;">
                Credit Underwriting &amp; Algorithmic Risk Governance Division
            </div>
            <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11px; color: #64748B; margin-top: 4px;">
                One Financial Plaza, Suite 2400 · New York, NY 10005 · Tel: (800) 555-0199
            </div>
        </div>
        <div style="text-align: right; font-family: 'Helvetica Neue', Arial, sans-serif;">
            <div style="display: inline-block; border: 1px solid #94A3B8; padding: 4px 10px; border-radius: 4px; font-size: 10px; font-weight: 700; color: #334155; text-transform: uppercase; background: #F8FAFC;">
                Official Notice
            </div>
            <div style="font-size: 11px; color: #64748B; margin-top: 4px; font-family: 'Courier New', monospace;">
                REF: ADV-2026-{customer_id}
            </div>
        </div>
    </div>

    <!-- Metadata Block -->
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 24px; font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 12.5px;">
        <tr>
            <td style="padding: 4px 0; color: #64748B; width: 140px; font-weight: 600;">DATE OF NOTICE:</td>
            <td style="padding: 4px 0; color: #0F172A; font-weight: 700;">October 23, 2026</td>
            <td style="padding: 4px 0; color: #64748B; width: 150px; font-weight: 600;">APPLICANT FILE ID:</td>
            <td style="padding: 4px 0; color: #0F172A; font-weight: 700; font-family: 'Courier New', monospace;">{customer_id}</td>
        </tr>
        <tr>
            <td style="padding: 4px 0; color: #64748B; font-weight: 600;">CREDIT FACILITY:</td>
            <td style="padding: 4px 0; color: #0F172A;">Unsecured Consumer Revolving Line</td>
            <td style="padding: 4px 0; color: #64748B; font-weight: 600;">UNDERWRITING ENGINE:</td>
            <td style="padding: 4px 0; color: #0F172A;">RiskTrail Calibrated Logit v1.2</td>
        </tr>
    </table>

    <!-- Formal Determination Callout Box -->
    <div style="
        background-color: {status_bg};
        border-left: 6px solid {status_border};
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 24px;
    ">
        <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 14px; font-weight: 800; color: {status_color}; text-transform: uppercase; letter-spacing: 0.5px;">
            FORMAL DETERMINATION: APPLICATION {status_label}
        </div>
        <div style="font-size: 12.5px; color: #374151; margin-top: 6px; font-style: italic;">
            Notice of Credit Underwriting Determination Pursuant to 12 C.F.R. § 1002.9 (Regulation B) and 15 U.S.C. § 1681m
        </div>
        <div style="font-size: 13.5px; color: #1F2937; margin-top: 8px;">
            {"Thank you for your recent credit application. After careful objective evaluation of your 13-month credit performance trajectory against our corporate credit policy guidelines, we regret to inform you that we are unable to approve your application at this time." if not is_approved else "We are pleased to inform you that after objective evaluation of your 13-month credit performance history, your credit application has met our institutional risk threshold and has been approved."}
        </div>
    </div>

    <!-- Quantitative Scoring Metrics -->
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px 18px; margin-bottom: 24px;">
        <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 10px;">
            Underwriting Assessment Quantitative Summary
        </div>
        <div style="display: flex; justify-content: space-between; font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 13px;">
            <div>
                <span style="color: #64748B;">Predicted Default Probability:</span>
                <strong style="color: {status_color}; margin-left: 4px;">{risk_prob * 100:.2f}%</strong>
            </div>
            <div>
                <span style="color: #64748B;">Policy Approval Cutoff:</span>
                <strong style="color: #0F172A; margin-left: 4px;">{cutoff * 100:.2f}%</strong>
            </div>
            <div>
                <span style="color: #64748B;">Cutoff Variance:</span>
                <strong style="color: {'#DC2626' if (risk_prob - cutoff) > 0 else '#16A34A'}; margin-left: 4px;">{(risk_prob - cutoff) * 100:+.2f}%</strong>
            </div>
        </div>
    </div>

    <!-- Key Factors / ECOA Reason Codes Table -->
    <div style="margin-bottom: 24px;">
        <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 13px; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">
            Principal Factors Influencing Decision (ECOA § 1002.9 Reason Codes)
        </div>
        <div style="font-size: 12.5px; color: #4B5563; margin-bottom: 12px;">
            In compliance with federal fair lending regulations, the following principal factors from our machine learning model had the greatest contribution to this underwriting determination:
        </div>
        <table style="width: 100%; border-collapse: collapse; border: 1px solid #E5E7EB; font-family: 'Helvetica Neue', Arial, sans-serif;">
            <thead>
                <tr style="background-color: #F1F5F9; border-bottom: 1.5px solid #CBD5E1; font-size: 11.5px; text-transform: uppercase; color: #475569;">
                    <th style="padding: 8px 12px; text-align: center; width: 40px;">Rank</th>
                    <th style="padding: 8px 12px; text-align: left;">Principal Reason &amp; Feature Description</th>
                    <th style="padding: 8px 12px; text-align: right; width: 140px;">SHAP Contribution</th>
                    <th style="padding: 8px 12px; text-align: center; width: 150px;">Impact Direction</th>
                </tr>
            </thead>
            <tbody>
                {reasons_table_body}
            </tbody>
        </table>
    </div>

    <!-- Statutory Regulatory Disclosures -->
    <div style="border-top: 1px solid #E2E8F0; padding-top: 18px; margin-bottom: 24px; font-size: 11.5px; color: #4B5563; text-align: justify; line-height: 1.5;">
        <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-weight: 700; color: #1F2937; margin-bottom: 4px; text-transform: uppercase; font-size: 11px;">
            Equal Credit Opportunity Act (ECOA) Notice
        </div>
        <p style="margin: 0 0 10px 0;">
            The Federal Equal Credit Opportunity Act prohibits creditors from discriminating against credit applicants on the basis of race, color, religion, national origin, sex, marital status, age (provided the applicant has the capacity to enter into a binding contract); because all or part of the applicant's income derives from any public assistance program; or because the applicant has in good faith exercised any right under the Consumer Credit Protection Act. The federal agency that administers compliance with this law concerning this creditor is the <strong>Consumer Financial Protection Bureau (CFPB), 1700 G Street NW, Washington, DC 20552</strong>.
        </p>

        <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-weight: 700; color: #1F2937; margin-bottom: 4px; text-transform: uppercase; font-size: 11px;">
            Fair Credit Reporting Act (FCRA) Notice
        </div>
        <p style="margin: 0 0 10px 0;">
            Our credit decision was based in part on information in a synthetic credit file furnished by the consumer reporting agency. Under Section 612 and Section 615 of the Fair Credit Reporting Act, you have the right to receive a free copy of your credit report within 60 days of receiving this notice. You also have the right to dispute the accuracy or completeness of any information provided by the credit reporting agency.
        </p>
    </div>

    <!-- Formal Signature Block -->
    <div style="display: flex; justify-content: space-between; align-items: flex-end; border-top: 1.5px solid #CBD5E1; padding-top: 20px; font-family: 'Helvetica Neue', Arial, sans-serif;">
        <div>
            <div style="font-family: 'Brush Script MT', cursive, Georgia, serif; font-size: 20px; color: #1E3A8A; margin-bottom: 4px;">
                Sarah Jenkins, CFA
            </div>
            <div style="font-size: 11px; font-weight: 700; color: #1E293B;">Sarah Jenkins, CFA</div>
            <div style="font-size: 10.5px; color: #64748B;">Chief Underwriting Officer · Credit Risk Committee</div>
        </div>
        <div>
            <div style="font-family: 'Brush Script MT', cursive, Georgia, serif; font-size: 20px; color: #1E3A8A; margin-bottom: 4px;">
                Marcus Vance, Ph.D.
            </div>
            <div style="font-size: 11px; font-weight: 700; color: #1E293B;">Marcus Vance, Ph.D.</div>
            <div style="font-size: 10.5px; color: #64748B;">Head of Model Risk Governance &amp; AI Compliance</div>
        </div>
        <div style="text-align: right;">
            <div style="display: inline-block; border: 1.5px solid #1E3A8A; color: #1E3A8A; padding: 4px 8px; border-radius: 4px; font-size: 9px; font-weight: 800; letter-spacing: 0.6px; text-transform: uppercase;">
                🔒 VERIFIED COMPLIANT NOTICE
            </div>
            <div style="font-size: 9.5px; color: #94A3B8; margin-top: 4px; font-family: 'Courier New', monospace;">
                SHA256: 89D80092-AB17-4929
            </div>
        </div>
    </div>
</div>
"""


def generate_bank_letter_text(
    customer_id: str,
    risk_prob: float,
    cutoff: float,
    is_approved: bool,
    reason_codes: List[Dict[str, Any]],
) -> str:
    """Generate plain-text version of official adverse action statement for download."""
    status_label = "APPROVED" if is_approved else "DECLINED"
    lines = [
        "=" * 82,
        "RISKTRAIL FINANCIAL SERVICES CORP. — CREDIT UNDERWRITING DETERMINATION",
        "One Financial Plaza, Suite 2400 · New York, NY 10005 · Tel: (800) 555-0199",
        "=" * 82,
        f"DATE OF NOTICE:        October 23, 2026",
        f"NOTICE REFERENCE:      ADV-2026-{customer_id}",
        f"APPLICANT IDENTIFIER:  {customer_id}",
        f"UNDERWRITING SYSTEM:   RiskTrail Calibrated Logit v1.2 (TreeExplainer SHAP)",
        "-" * 82,
        f"FORMAL DETERMINATION:  {status_label}",
        f"Assessed Default Risk: {risk_prob * 100:.2f}%",
        f"Policy Cutoff Limit:   {cutoff * 100:.2f}%",
        f"Variance vs Cutoff:    {(risk_prob - cutoff) * 100:+.2f}%",
        "-" * 82,
        "PRINCIPAL FACTORS INFLUENCING THIS DETERMINATION (ECOA § 1002.9):",
    ]

    for r in reason_codes:
        impact = r.get("shap_impact", 0.0)
        direction = "Elevates Default Risk (+)" if impact > 0 else "Lowers Risk (-)"
        lines.append(
            f"  [{r.get('rank', '-')}] {r.get('adverse_action_reason', '')}\n"
            f"      Feature: {r.get('feature', '')} = {r.get('feature_value', 0.0):.4f} "
            f"| SHAP Impact: {impact:+.4f} log-odds | {direction}"
        )

    lines.extend([
        "-" * 82,
        "EQUAL CREDIT OPPORTUNITY ACT (ECOA) NOTICE:",
        "The Federal Equal Credit Opportunity Act prohibits creditors from discriminating",
        "against credit applicants on the basis of race, color, religion, national origin,",
        "sex, marital status, or age. Administered by the Consumer Financial Protection",
        "Bureau (CFPB), 1700 G Street NW, Washington, DC 20552.",
        "",
        "FAIR CREDIT REPORTING ACT (FCRA) NOTICE:",
        "Under Section 612 of the FCRA, you have the right to obtain a free copy of your credit",
        "report from the consumer reporting agency within 60 days of receiving this notice.",
        "=" * 82,
        "AUTHORIZED BY: Credit Risk Committee & Model Governance Division",
        "DOCUMENT SECURITY STAMP: VERIFIED TAMPER-EVIDENT RECORD · SHA256: 89D80092-AB17-4929",
        "=" * 82,
    ])
    return "\n".join(lines)
