fn evaluate(text) {
    // Defensive per the provider contract: a provider may be evaluated
    // for ANY decision event, so any argument may be absent (unit).
    // Return a conforming sentinel instead of erroring (errors are UB).
    if type_of(text) == "()" {
        return #{ severityScore: parse_decimal("-1.0") };
    }

    let score = if text == "safe" {
        parse_decimal("0.10")
    } else if text == "spam" {
        parse_decimal("0.80")
    } else {
        parse_decimal("0.50")
    };
    #{ severityScore: score }
}
