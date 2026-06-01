"""Usage examples for domain hallucination guard."""

import asyncio
import json
from pathlib import Path

from . import actions, config, kb


async def example_basic_detection():
    """Basic domain hallucination detection example."""
    print("=" * 60)
    print("Example 1: Basic Detection")
    print("=" * 60)

    answer = """
    PyTorch is a popular deep learning framework. You can find more information at:
    - Official site: https://pytorch.org
    - GitHub repo: https://github.com/pytorch/pytorch
    - Documentation: https://pytorch.readthedocs.io
    """

    result = await actions.analyze_answer(
        answer=answer,
        user_query="What is PyTorch?",
        verification_level="dns",
    )

    print(f"Status: {result['status']}")
    print(f"Has Issues: {result['detection']['has_issues']}")
    print(f"Risk Score: {result['risk_score']['score']}")
    print(f"Decision: {result['decision']['action']}")
    print()


async def example_with_hallucination():
    """Example with a hallucinated domain."""
    print("=" * 60)
    print("Example 2: With Hallucinated Domain")
    print("=" * 60)

    answer = """
    To install PyTorch, visit https://pytorch-official.xyz
    and download from https://github.com/pytorch-team/pytorch-fork
    """

    result = await actions.analyze_answer(
        answer=answer,
        user_query="How do I install PyTorch?",
        verification_level="dns",
    )

    print(f"Status: {result['status']}")
    if result['detection']['has_issues']:
        print(f"Issues Found: {len(result['detection']['issues'])}")
        for issue in result['detection']['issues']:
            print(f"  - {issue['type']}: {issue['target']} (severity: {issue['severity']})")
    print(f"Risk Score: {result['risk_score']['score']}")
    print(f"Decision: {result['decision']['action']}")
    if result['decision']['action'] != 'pass':
        print(f"Modified Answer: {result['enforced_answer']['modified_answer'][:100]}...")
    print()


async def example_no_links():
    """Example with no external links (fast pass)."""
    print("=" * 60)
    print("Example 3: Fast Pass (No Links)")
    print("=" * 60)

    answer = "Machine learning is a subset of artificial intelligence."

    result = await actions.analyze_answer(
        answer=answer,
        user_query="What is machine learning?",
        verification_level="dns",
    )

    print(f"Status: {result['status']}")
    print(f"Reason: {result.get('reason', 'N/A')}")
    print(f"Decision: {result['decision']['action']}")
    print()


async def example_with_kb():
    """Example with custom knowledge base."""
    print("=" * 60)
    print("Example 4: With Custom Knowledge Base")
    print("=" * 60)

    # Initialize KB with seed data
    kb_instance = kb.initialize_kb()
    kb_instance.add_trusted_domain("my-custom-domain.com")
    kb_instance.add_blacklisted_domain("phishing-site.com", reason="Known phishing")

    answer = """
    You can learn more at https://my-custom-domain.com
    or visit https://phishing-site.com (don't visit this!)
    """

    result = await actions.analyze_answer(
        answer=answer,
        user_query="Where can I learn more?",
        verification_level="dns",
        kb_instance=kb_instance,
    )

    print(f"Status: {result['status']}")
    if result['detection']['has_issues']:
        print(f"Issues Found: {len(result['detection']['issues'])}")
        for issue in result['detection']['issues']:
            print(f"  - {issue['type']}: {issue['target']}")
    print(f"Risk Score: {result['risk_score']['score']}")
    print(f"Decision: {result['decision']['action']}")
    print()


async def example_configuration():
    """Example showing configuration management."""
    print("=" * 60)
    print("Example 5: Configuration Management")
    print("=" * 60)

    # Create custom config
    cfg = config.DomainHallucinationGuardConfig(
        verification=config.VerificationConfig(level="dns"),
        detection=config.DetectionConfig(
            enable_semantic_check=False,
            enable_advanced_verification=False,
            no_link_fast_pass=True,
        ),
        scoring=config.ScoringConfig(
            fail_threshold=60.0,
            refine_threshold=40.0,
            warn_threshold=20.0,
        ),
    )

    print("Current Configuration:")
    print(json.dumps(cfg.to_dict(), indent=2))

    # Save and load
    config_path = Path("/tmp/test_config.json")
    cfg.save(config_path)
    print(f"\nConfiguration saved to {config_path}")

    loaded_cfg = config.DomainHallucinationGuardConfig.load(config_path)
    print(f"Configuration loaded. Verification level: {loaded_cfg.verification.level}")
    print()


async def example_batch_analysis():
    """Example analyzing multiple answers."""
    print("=" * 60)
    print("Example 6: Batch Analysis")
    print("=" * 60)

    answers = [
        "Visit https://github.com/pytorch/pytorch for more info.",
        "Go to https://fake-pytorch-repo.xyz for installation.",
        "Machine learning is a subset of AI.",
        "Check out https://example.com and https://another.org",
    ]

    results = []
    for i, answer in enumerate(answers, 1):
        result = await actions.analyze_answer(
            answer=answer,
            user_query="Learn about PyTorch",
            verification_level="dns",
        )
        results.append(result)
        print(f"Answer {i}: {result['status']}")
        print(f"  Risk Score: {result.get('risk_score', {}).get('score', 'N/A')}")
        print(f"  Decision: {result['decision']['action']}")

    print(f"\nProcessed {len(results)} answers")
    passed = sum(1 for r in results if r['decision']['action'] == 'pass')
    print(f"  Passed: {passed}")
    print(f"  Flagged: {len(results) - passed}")
    print()


async def example_custom_policy():
    """Example with custom scoring policy."""
    print("=" * 60)
    print("Example 7: Custom Policy")
    print("=" * 60)

    from . import decision as decision_module

    # Simulate a detection result
    detection = {
        "has_issues": True,
        "issues": [
            {
                "type": "non_existent_domain",
                "target": "fake-domain.com",
                "severity": "high",
            }
        ],
        "issue_summary": {"total": 1, "by_type": {"non_existent_domain": 1}},
    }

    from . import scoring as scoring_module
    risk_score = scoring_module.calculate_risk_score(detection)

    # Create custom policy with stricter thresholds
    policy = decision_module.PolicyThresholds(
        fail_threshold=40.0,  # Lower threshold = stricter
        refine_threshold=20.0,
        warn_threshold=10.0,
    )

    enforcement = decision_module.make_decision(
        risk_score=risk_score,
        policy=policy,
        verification_level="dns",
    )

    print(f"Risk Score: {risk_score['score']}")
    print(f"Threshold (fail): {policy.fail_threshold}")
    print(f"Decision: {enforcement['action']}")
    print(f"Reason: {enforcement['reason']}")
    print()


async def main():
    """Run all examples."""
    print("\nDomain Hallucination Guard - Usage Examples\n")

    await example_basic_detection()
    await example_with_hallucination()
    await example_no_links()
    await example_with_kb()
    await example_configuration()
    await example_batch_analysis()
    await example_custom_policy()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
