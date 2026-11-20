// Vendor::ThreatScan(text) -> { severity: Vendor::Severity }.
// Returns the bare enum id; Dogwood converts it to the enum entity and denies
// the request if the id isn't one of the declared values.
fn evaluate(text) {
    // A provider may be evaluated for ANY decision event, so the argument may
    // be absent (unit). Return the most restrictive value.
    if type_of(text) == "()" { return #{ severity: "CRITICAL" }; }
    let t = to_lower(text);
    if t.contains("ransomware") { #{ severity: "CRITICAL" } }
    else if t.contains("malware") { #{ severity: "HIGH" } }
    else if t.contains("scan") { #{ severity: "MEDIUM" } }
    else if t.contains("garbled") { #{ severity: "SEVERE" } }   // not a declared value
    else { #{ severity: "LOW" } }
}
