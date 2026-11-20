// `Content::Filter(text, categories) -> { <CATEGORY>: { severityScore } }`.
//
// Takes a document and a SET of category names (the second declared
// argument, `paramType: set`, arrives as a Rhai array). Returns a record
// with every category that the output type declares (VIOLENCE and HATE),
// each with a decimal `severityScore`, so the policy can project
// `["VIOLENCE"].severityScore` and compare it. Only the categories in the
// set get a real score. Scores here are a fixed lookup per keyword; a real
// provider would call a classifier.
fn score_for(text, category) {
    if text == "violent" && category == "VIOLENCE" {
        parse_decimal("0.90")
    } else if text == "hateful" && category == "HATE" {
        parse_decimal("0.90")
    } else {
        parse_decimal("0.10")
    }
}

fn evaluate(text, categories) {
    // Defensive per the provider contract: a provider may be evaluated
    // for ANY decision event, so any argument may be absent (unit).
    // Return a conforming sentinel instead of erroring (errors are UB).
    if type_of(text) == "()" || type_of(categories) == "()" {
        return #{ VIOLENCE: #{ severityScore: parse_decimal("-1.0") }, HATE: #{ severityScore: parse_decimal("-1.0") } };
    }

    // The output type requires every declared category. A category that is
    // not in the set gets the score 1.0, so a policy that reads it sees a
    // high severity instead of a low one.
    let out = #{
        VIOLENCE: #{ severityScore: parse_decimal("1.0") },
        HATE: #{ severityScore: parse_decimal("1.0") }
    };
    for category in categories {
        // The script ignores an undeclared category.
        if out.contains(category) {
            out[category] = #{ severityScore: score_for(text, category) };
        }
    }
    out
}
