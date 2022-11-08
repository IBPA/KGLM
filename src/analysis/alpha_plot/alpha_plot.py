import matplotlib
matplotlib.use("Agg")

from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    results_files = glob('./*.txt')
    print(results_files)

    mr = [41.35226547543076,
          40.828334396936825,
          40.45660497766433,
          40.22878111040204,
          40.157147415443525,
          40.18395022335673,
          40.359763880025525,
          40.692724952137844,
          41.15555201021059,
          41.74074664964901,
          42.49920229738353
    ]

    mrr = [0.45889052518838835,
          0.4633706963294484,
          0.4656842251986144,
          0.46633558605599834,
          0.4667899305918577,
          0.4668339720831109,
          0.46595005460438116,
          0.46540014373742605,
          0.4630248000627723,
          0.46069078429231236,
          0.4557007044699313
    ]

    hits1 = [0.32275047862156986,
          0.3270580727504786,
          0.3297702616464582,
          0.3296107211231653,
          0.3299298021697511,
          0.3296107211231653,
          0.3289725590299936,
          0.32929164007657946,
          0.3265794511805999,
          0.32514358647096364,
          0.31940012763241865
    ]

    hits3 = [0.5320676451818762,
          0.5384492661135929,
          0.5397255902999362,
          0.541799617102744,
          0.5392469687300574,
          0.5376515634971283,
          0.5382897255903,
          0.5368538608806637,
          0.5322271857051691,
          0.5309508615188258,
          0.5237715379706446
    ]

    hits10 = [0.7281429483088705,
          0.7311742182514359,
          0.7358008934269304,
          0.7381940012763242,
          0.7402680280791321,
          0.7410657306955967,
          0.7401084875558391,
          0.7381940012763242,
          0.7369176770899809,
          0.7353222718570517,
          0.7324505424377792
    ]

    mr_difference = max(mr) - min(mr)
    mr_percent = mr_difference * 100 / max(mr)
    print(f'mr_percent: {mr_percent}, mr_difference: {mr_difference}')

    mrr_difference = max(mrr) - min(mrr)
    mrr_percent = mrr_difference * 100 / min(mrr)
    print(f'mrr_percent: {mrr_percent}, mrr_difference: {mrr_difference}')

    hits1_percent = (max(hits1) - min(hits1)) * 100 / min(hits1)
    print(f'hits1_percent: {hits1_percent}')

    hits3_percent = (max(hits3) - min(hits3)) * 100 / min(hits3)
    print(f'hits3_percent: {hits3_percent}')

    hits10_percent = (max(hits10) - min(hits10)) * 100 / min(hits10)
    print(f'hits10_percent: {hits10_percent}')

if __name__ == '__main__':
    main()
