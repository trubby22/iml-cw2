import pandas as pd


def f(x: str):
    print(x, type(x))

    foo = x[1:-1].replace(',', '-')
    print(foo)
    return foo


def transform_test_results_csv():
    df = pd.read_csv('hyper-parameter-results-raw.csv')
    test_loss = 'Test Loss'
    training_loss = 'Training Loss'
    df = df[[
        'batch_size',
        'learning_rate',
        'num_epochs',
        'shape',
        test_loss,
        training_loss
    ]]
    new_metric = 'Relative test and training loss difference'
    df[new_metric] = (df[test_loss] - df[training_loss]) / df[training_loss]
    df.sort_values(by=[test_loss, new_metric])
    relevant_columns = [test_loss, training_loss]
    for x in relevant_columns:
        df[x] = df[x].apply(lambda x: f'{x:_.0f}')
        # df[x] = df[x].apply(lambda x: f'{x:.0f}')
    df[new_metric] = df[new_metric].apply(lambda x: f'{x:.2f}')
    df['shape'] = df['shape'].apply(f)
    df.index = df.index + 1
    df.index.name = 'Rank'
    df.to_csv('hyper-parameter-results.csv')
    print(df)


if __name__ == '__main__':
    transform_test_results_csv()
