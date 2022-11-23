import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from src.visualization.visualize import ModelPredictionVisualization
from src.models.utils import load_model
from src.data.synthetic_dataset import SyntheticDataset


@click.command()
@click.argument('dataset', type=click.Path())
@click.argument('model_path', type=click.Path())
@click.argument('save_path', type=click.Path())
@click.option(
    '--n_viz',
    type=click.INT,
    default=4,
    help="how many visualizations to display at once. Has to be square number")
def main(dataset, model_path, save_path, n_viz):
    """This script creates a simple visualization of four randomly chosen predictions from DATASET. For normalization
    it uses the scaler that was used for training. It requires a pre-trained model with path MODEL_PATH and
    saves the predictions to SAVE_PATH.pdf.
    """
    model = load_model(model_path)
    dataset = SyntheticDataset.load(dataset)
    if not dataset.is_normalized:
        dataset.normalize(scaler=model.train_dataset.scaler)

    test_samples = np.random.randint(0, len(dataset), size=n_viz)
    inp_samples, outp_actual, characteristics = dataset[test_samples]

    inp_len = len(inp_samples[0])
    outp_pred = []
    for i, inp_sample in enumerate(inp_samples):
        sample_seq_len = characteristics[0][i, 2]
        out = model(inp_sample.reshape(1, -1, 1), sample_seq_len)
        outp_pred.append(out.to("cpu").detach().numpy().squeeze())
    outp_pred = np.array(outp_pred)
    outp_actual = outp_actual.reshape(n_viz, inp_len)
    inp_samples = inp_samples.reshape(n_viz, inp_len)

    # inverse normalization
    inp_samples = dataset.inverse_normalization(data=inp_samples)
    outp_actual = dataset.inverse_normalization(data=outp_actual)
    outp_pred = dataset.inverse_normalization(data=outp_pred)

    import os  #TODO, this should only be a temporal workaround
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # visualize the predictions

    viz = ModelPredictionVisualization(use_latex_font=False)
    viz.subplots(nrows=int(np.sqrt(n_viz)),
                 ncols=int(np.sqrt(n_viz)),
                 figsize=(16, 16))
    viz.plot(inp_samples,
             outp_actual,
             outp_pred,
             pred_starts=np.ones(16, dtype=int) * inp_len)
    viz.fig.suptitle("Different predictions for test samples")
    viz.plt.legend()
    viz.save(filename=save_path)
    viz.plt.show()


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
