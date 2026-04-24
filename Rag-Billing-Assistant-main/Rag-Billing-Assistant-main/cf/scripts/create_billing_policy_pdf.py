from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def build_billing_policy_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
        title="Billing Policy - Source of Truth",
        author="SaaS Billing & Refund Assistant",
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h1_style = styles["Heading1"]
    h2_style = styles["Heading2"]
    body_style = styles["BodyText"]

    story = []

    story.append(Paragraph("Billing Policy Source of Truth", title_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Document Purpose", h1_style))
    story.append(
        Paragraph(
            "This document is the authoritative policy reference for billing, refunds, and "
            "human review decisions in the SaaS Billing and Refund Assistant.",
            body_style,
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Subscription Tiers", h1_style))

    story.append(Paragraph("Basic Tier", h2_style))
    story.append(Paragraph("Price: $10 per month.", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Pro Tier", h2_style))
    story.append(Paragraph("Price: $50 per month.", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Enterprise Tier", h2_style))
    story.append(Paragraph("Price: $200 per month.", body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Refund Rules", h1_style))
    story.append(Paragraph("Rule 1: 100% refund within 7 days of payment.", body_style))
    story.append(Paragraph("Rule 2: No refunds after 14 days of payment.", body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Human Review Rule", h1_style))
    story.append(
        Paragraph(
            "Any refund request for Enterprise tier subscriptions must be approved by a human manager.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Any refund request with an amount greater than $100 must be approved by a human manager.",
            body_style,
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Failed Payments", h1_style))
    story.append(
        Paragraph(
            "Account suspension happens after 3 failed payment attempts.",
            body_style,
        )
    )

    doc.build(story)


def main() -> None:
    output_pdf = Path("data/raw_docs/billing_policy.pdf")
    build_billing_policy_pdf(output_pdf)
    print(f"Created PDF at: {output_pdf.resolve()}")


if __name__ == "__main__":
    main()
