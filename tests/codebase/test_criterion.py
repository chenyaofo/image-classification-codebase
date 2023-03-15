from codebase.criterion import CRITERION


def test_criterion():
    CRITERION.build_from(
        dict(
            type_="LabelSmoothCrossEntropyLoss",
            num_classes=1000
        )
    )
