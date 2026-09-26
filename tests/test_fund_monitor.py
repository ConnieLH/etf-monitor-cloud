import unittest
from decimal import Decimal, localcontext

import pandas as pd

from fund_monitor import compute_metrics


class FrFormulaTests(unittest.TestCase):
    def test_fr_matches_manual_recursive_ema_calculation(self):
        closes = [10, 11, 9, 12, 13, 8, 10, 14, 12, 15, 11, 9,
                  10, 13, 16, 12, 11, 14, 17, 15, 13, 10, 12, 16,
                  18, 14, 13, 17, 19, 15, 12, 16, 20, 18, 14, 17]
        result = compute_metrics(pd.DataFrame({"close": closes}))
        # Hand recurrence, independent of pandas ewm: seed E[0] = close[0],
        # then E[t] = alpha * close[t] + (1-alpha) * E[t-1].
        with localcontext() as context:
            context.prec = 40
            ema = {span: Decimal(closes[0]) for span in (5, 10, 12, 26)}
            previous_fr = None
            for i, close in enumerate(closes):
                if i:
                    for span in ema:
                        alpha = Decimal(2) / Decimal(span + 1)
                        ema[span] = alpha * Decimal(close) + (1 - alpha) * ema[span]
                fr1 = ema[12] - ema[26]
                fr = fr1 / ema[5]
                expected = {"ema_5": ema[5], "ema_10": ema[10], "fr1": fr1, "fr": fr}
                if previous_fr is None:
                    self.assertTrue(pd.isna(result.loc[i, "fr_bar"]))
                else:
                    expected["fr_bar"] = (fr - previous_fr) * 3
                for column, value in expected.items():
                    with self.subTest(row=i, column=column):
                        self.assertLess(abs(result.loc[i, column] - float(value)), 1e-9)
                previous_fr = fr


if __name__ == "__main__":
    unittest.main()
