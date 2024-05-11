#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# just trying to see how log scale and ylim getting/setting interact, for use in
# al_analysis sensitivity analysis plots

def main():
    #xs = np.linspace(-1, 3, num=5)
    #ys = [10**x for x in xs]
    #xs = [1, 2, 3]
    #ys = [0, 1, 2]
    xs = [1, 2, 3, 4]
    ys = [0, 1, 2, 0]

    df = pd.DataFrame({'x': xs, 'y': ys})

    # https://datascience.stackexchange.com/questions/89850
    #df['y'] = df['y'] + 1/10

    fig, ax = plt.subplots()

    sns.lineplot(df.y, ax=ax)

    # default 'clip' apparently (not sure if i can control clip value?)
    nonpositive = 'mask'
    #nonpositive = 'clip'
    print(f'{nonpositive=}')
    print(f'before ax.set_yscale("log"): {ax.get_ylim()=}')
    #ax.set_yscale('log')
    ax.set_yscale('log', nonpositive=nonpositive)
    print(f'AFTER ax.set_yscale("log"): {ax.get_ylim()=}')

    #ax.set_ylim(0, 1)
    #ax.set_ylim(1/10, df.y.max())

    plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

