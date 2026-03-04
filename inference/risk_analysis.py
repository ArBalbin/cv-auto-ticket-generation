def analyze_risk(persons, width, height):
    count = len(persons)

    # Simple density calculation
    area = width * height
    density = count / area * 100000  # scaled density

    if count >= 4:
        level = "HIGH"
    elif count >= 2:
        level = "MEDIUM"
    elif count > 0:
        level = "LOW"
    else:
        level = "NONE"

    # if count >= 1:
    #     level = "HIGH"
    # else:
    #     level = "LOW"

    return {
        "count": count,
        "level": level,
        "avg_density": density,
        "max_density": density
    }
