# Credit data (Chapter 12, scoring models)

Two public credit data sets, downloaded once from the UCI Machine Learning Repository on 3 October 2026 and saved here.
The code reads them with `load_credit(name)` (Quantlets/Ch_12/generate_all_charts.py): local file first, otherwise
`https://raw.githubusercontent.com/danpele/SFM/main/data/credit/<file>`.

| File | Data set | Source | Licence |
|---|---|---|---|
| `south_german_credit.csv` | South German Credit: 1000 consumer credits of a regional bank in southern Germany, 1973–1975; 300 bad, 700 good; 20 predictors | UCI, https://doi.org/10.24432/C5QG88 (file `SouthGermanCredit.asc`) | CC BY 4.0 |
| `south_german_credit_codetable.txt` | Code table of the South German Credit variables | the same archive | CC BY 4.0 |
| `taiwan_credit_card_default.csv` | Default of Credit Card Clients: 30,000 card holders in Taiwan, April–September 2005; default on the October 2005 payment | UCI, https://doi.org/10.24432/C55S3H (file `default of credit card clients.xls`) | CC BY 4.0 |

## Changes made when saving

- South German Credit: the German column names of the original file (laufkont, laufzeit, ...) are replaced by the
  English names of Grömping (2019) (status, duration, credit_history, ...); the codes are unchanged.
  `credit_risk` = 1 for a good credit, 0 for a bad one; the code adds `default = 1 - credit_risk`.
- Taiwan: the header row of the Excel file is used; the last column `default payment next month` is renamed `default`;
  saved as CSV. No other change.

## Notes

- South German Credit is the corrected version of the UCI "Statlog (German Credit Data)" (https://doi.org/10.24432/C5NC77),
  whose code labels are wrong for several variables (e.g. the checking account and the credit history).
- Bad credits are heavily oversampled: the bank's bad rate was about 5% (Grömping, 2019). All borrowers had passed the
  bank's checks, so rejected applicants are missing.

## References

- Grömping, U. (2019). South German credit data: correcting a widely used data set. Reports in Mathematics, Physics and
  Chemistry 4/2019, Beuth University of Applied Sciences Berlin. http://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf
- South German Credit [data set] (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5QG88
- Yeh, I.-C., Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of
  default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
- Default of Credit Card Clients [data set] (2009). UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
