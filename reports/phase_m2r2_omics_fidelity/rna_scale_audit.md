# RNA scale audit

## Source metadata

- The source profile metadata identifies `data_mrna_seq_v2_rsem.txt` as RSEM batch-normalized mRNA expression from Illumina HiSeq RNASeqV2.
- Negative values are retained as normalized continuous values. They are not treated as a missing-value sentinel.

## Observed values

- min=-0.9641786217689514; median=219.72862243652344; q99=11151.0; max=894409.0; finite_count=9898000; negative_fraction=0.01689088704788846.
- Negative audit: count=167186; unique_values=9305; values_leq_minus_one=0; most_common=[{'count': 191, 'value': -0.22322656214237213}, {'count': 186, 'value': -0.028129147365689278}, {'count': 185, 'value': -0.15483000874519348}, {'count': 185, 'value': -0.04935970902442932}, {'count': 183, 'value': -0.36356469988822937}, {'count': 182, 'value': -0.3034227788448334}, {'count': 180, 'value': -0.14276665449142456}, {'count': 178, 'value': -0.506978452205658}, {'count': 178, 'value': -0.22026583552360535}, {'count': 177, 'value': -0.1390620917081833}].
- Anonymous gene/value examples: [{'gene': '', 'value': -0.26337236166000366}, {'gene': 'A2ML1', 'value': -0.3211534023284912}, {'gene': 'AACSL', 'value': -0.2045004516839981}, {'gene': 'ABCA12', 'value': -0.27125778794288635}, {'gene': 'ABCA17P', 'value': -0.2387361079454422}].

## Decision

- Apply metadata-verified `log2(x+1)` without clipping the negative tail. All observed finite values are above -1; any finite value <= -1 fails closed.
- Imputation, variance ranking, signature intersection and standardization are fit on the outer training fold. This correction required rerunning B0-B6.
