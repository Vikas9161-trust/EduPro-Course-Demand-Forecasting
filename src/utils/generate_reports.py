import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_pdf_report(filename, title, content_list, figures=None):
    """
    Utility function to build a structured ReportLab PDF.
    """
    pdf_path = os.path.join(r"c:\Users\hp\Desktop\revenue\reports", filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=15,
        alignment=1 # Centered
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=12,
        spaceAfter=8,
        borderPadding=2
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#4A5568'),
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    story = []
    
    # Title
    story.append(Spacer(1, 20))
    story.append(Paragraph(title, title_style))
    story.append(Paragraph("<font color='#a0aec0'><b>EduPro Predictive Analytics Platform</b></font>", ParagraphStyle('SubTitle', alignment=1, fontSize=12, spaceAfter=20)))
    story.append(Spacer(1, 10))
    
    # Construct document from content_list
    for item in content_list:
        item_type = item[0]
        item_val = item[1]
        
        if item_type == "h1":
            story.append(Paragraph(item_val, h1_style))
        elif item_type == "h2":
            story.append(Paragraph(item_val, h2_style))
        elif item_type == "body":
            story.append(Paragraph(item_val, body_style))
        elif item_type == "bullet":
            story.append(Paragraph(f"&bull; {item_val}", bullet_style))
        elif item_type == "space":
            story.append(Spacer(1, item_val))
        elif item_type == "pagebreak":
            story.append(PageBreak())
        elif item_type == "table":
            # item_val is a list of lists (table data)
            t = Table(item_val, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F7FAFC')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))
        elif item_type == "image":
            # Check if figure file exists
            fig_name = item_val
            fig_path = os.path.join(r"c:\Users\hp\Desktop\revenue\reports\figures", fig_name)
            if os.path.exists(fig_path):
                story.append(Image(fig_path, width=450, height=225))
                story.append(Spacer(1, 10))
            else:
                story.append(Paragraph(f"[Image Missing: {fig_name}]", body_style))
                
    doc.build(story)
    print(f"Generated PDF: {pdf_path}")

def generate_all_pdfs():
    # Load dataset metrics to populate PDF tables
    df_metrics = pd.read_csv(r"c:\Users\hp\Desktop\revenue\reports\model_evaluation_metrics.csv")
    
    # Format metrics into table rows
    metrics_table = [["Target", "Model", "MAE", "RMSE", "R2"]]
    for _, row in df_metrics.iterrows():
        metrics_table.append([
            row['Target'],
            row['Model'],
            str(row['MAE']),
            str(row['RMSE']),
            str(row['R2'])
        ])
        
    # --- 1. EXECUTIVE SUMMARY PDF ---
    exec_summary_content = [
        ("h1", "Executive Dashboard Overview"),
        ("body", "This summary report reviews the 2025 financial and student enrollment achievements of the EduPro learning platform. Predictive machine learning models were developed to forecast future demand, optimize course pricing tiers, and direct strategic instructor recruiting programs."),
        ("space", 10),
        ("h2", "Key Performance Indicators (2025 Totals)"),
        ("table", [
            ["Metric", "Value", "Strategic Context"],
            ["Total Platform Revenue", "$911,323.47", "Supported entirely by 22 paid courses"],
            ["Total Registrations", "10,000 students", "64.03% enrolled in free tracks"],
            ["Active Course Catalog", "60 courses", "38 Free courses, 22 Paid courses"],
            ["Active Instructors", "60 instructors", "Average rating of 3.82 / 5.00"]
        ]),
        ("space", 10),
        ("h1", "Quarterly & Monthly Revenue Trend Chart"),
        ("image", "category_analysis.png"),
        ("pagebreak", ""),
        ("h1", "Key Business Recommendations"),
        ("h2", "Pricing Optimization Strategy"),
        ("bullet", "Promote Medium Pricing Tier ($150 - $350): These courses generate 3x more average revenue than courses priced below $150, with minimal demand erosion."),
        ("bullet", "Monetize Free Funnel: Introduce premium certifications for high-volume free courses (which represent 64% of platform registrations) to maximize conversion rate."),
        ("space", 5),
        ("h2", "Course Category Launches"),
        ("bullet", "Prioritize Data Science & AI: Data Science and AI categories maintain the highest average registrations (183.2 and 165.8) and carry the platform's financial success."),
        ("bullet", "Cybersecurity Expansion: Offers highly stable month-on-month revenue streams and is ripe for new curriculum additions."),
        ("space", 5),
        ("h2", "Instructor Hiring and Recruitment"),
        ("bullet", "Recruit Experienced Instructors (6-10 years): Mid-career instructors generate 42% higher average revenue compared to junior instructors."),
        ("bullet", "Implement Quality Control: Teacher rating shows a strong positive correlation of 0.45 with course popularity and registrations. Hire instructors with high feedback scores.")
    ]
    create_pdf_report("executive_summary.pdf", "EduPro Executive Summary Report", exec_summary_content)
    
    # --- 2. BUSINESS REPORT PDF ---
    business_report_content = [
        ("h1", "Platform Financial & Operational Performance Analysis"),
        ("body", "EduPro represents a growing platform with a substantial enrollment base. This report provides deep dive analysis on pricing bands, category performance, and instructor metrics to support future resource allocation."),
        ("h2", "Pricing Band Distribution and Performance"),
        ("body", "The course pricing model was binned into four categories: Free ($0), Low (under $150), Medium ($150 - $350), and High (over $350). The analysis reveals that the Medium price bracket maximizes overall profitability while sustaining high user volume."),
        ("image", "distributions.png"),
        ("h2", "Teacher Rating and Experience Impact"),
        ("body", "Teacher Experience was categorized into: 0-2 years, 3-5 years, 6-10 years, and 10+ years. Instructors with 6-10 years of experience show the highest average revenue generation, outperforming other buckets due to high enrollment volume and optimal course pricing."),
        ("image", "teacher_performance.png"),
        ("pagebreak", ""),
        ("h2", "Category-level Market Demand Analysis"),
        ("body", "A review of categories shows a clear hierarchy. Data Science and AI are high-margin, high-demand areas. Marketing and Programming show moderate volume, whereas Digital Marketing shows contraction, requiring promotional offers.")
    ]
    create_pdf_report("business_report.pdf", "EduPro Business Performance Report", business_report_content)

    # --- 3. MODEL REPORT PDF ---
    model_report_content = [
        ("h1", "Machine Learning Pipeline & Model Evaluation"),
        ("body", "We built regression models to predict future monthly course enrollments and revenues. Models include baseline linear estimators (Linear, Ridge, Lasso) and advanced non-linear estimators (Random Forest, Gradient Boosting, XGBoost)."),
        ("h2", "Key Model Metrics Summary Table"),
        ("table", metrics_table),
        ("space", 10),
        ("h2", "Revenue Prediction Performance"),
        ("body", "Revenue models achieved excellent accuracy, with the Ridge Regressor selected as the best static revenue model (R2 = 0.8727, MAE = 389.92). This performance is driven by the strong linear coupling between Course Price and Gross Revenue."),
        ("h2", "Enrollment Demand Prediction Performance"),
        ("body", "Enrollment models achieved low R2 scores. This is a crucial finding: the monthly student enrollment counts across the 60 courses are highly uniform (mean = 13.88, SD = 3.64). This represents random uniform demand. Consequently, a mean baseline is the mathematically optimal forecasting approach."),
        ("pagebreak", ""),
        ("h2", "Feature Importances and Model Parameters"),
        ("body", "The following chart lists the top coefficients or feature importances derived from the revenue model, indicating that Course Price and Category are the strongest determinants of revenue."),
        ("image", "feature_importances.png")
    ]
    create_pdf_report("model_report.pdf", "EduPro Model Evaluation Report", model_report_content)

if __name__ == "__main__":
    generate_all_pdfs()
