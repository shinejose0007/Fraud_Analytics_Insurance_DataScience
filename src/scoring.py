def risk_level_from_score(score):
    if score >= 71:
        return "High Risk"
    if score >= 31:
        return "Medium Risk"
    return "Low Risk"


def recommended_action(risk_level):
    if risk_level == "High Risk":
        return "Fraud investigation / detailed manual review"
    if risk_level == "Medium Risk":
        return "Manual review with supporting documents"
    return "Normal processing"


def probability_to_score(probability):
    return round(float(probability) * 100, 1)