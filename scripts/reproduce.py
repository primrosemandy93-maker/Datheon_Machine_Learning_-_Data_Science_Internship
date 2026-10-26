"""Run the notebooks' Python cells in fresh processes and check measured results.

The notebooks use ordinary Python (no IPython magics). This runner captures
stdout, display tables, and Matplotlib figures without opening a Jupyter server.
"""
import argparse
import base64
import copy
from contextlib import redirect_stdout, redirect_stderr
from io import BytesIO, StringIO
from pathlib import Path
import os
import re
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    'iris': ('task_1_iris', 'Task_1_Iris_Classification.ipynb'),
    'housing': ('task_2_housing', 'Task_2_Housing_Price_Prediction.ipynb'),
}


def execute_notebook(source, output, update):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    namespace = {'__name__': '__main__'}
    outputs = []

    def display(value):
        data = {'text/plain': repr(value)}
        if isinstance(value, (pd.DataFrame, pd.Series)):
            frame = value.to_frame() if isinstance(value, pd.Series) else value
            data['text/html'] = frame.to_html()
        outputs.append(nbformat.v4.new_output('display_data', data=data))

    def show(*args, **kwargs):
        for num in plt.get_fignums():
            buffer = BytesIO()
            plt.figure(num).savefig(buffer, format='png', bbox_inches='tight')
            outputs.append(nbformat.v4.new_output('display_data', data={
                'image/png': base64.b64encode(buffer.getvalue()).decode('ascii'),
                'text/plain': '<Matplotlib figure>',
            }))
        plt.close('all')

    namespace['display'] = display
    plt.show = show
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        outputs = []
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(cell.source, f'{source.name}:cell-{count}', 'exec'), namespace)
        # Tables/figures are captured; streams are grouped per cell.
        streams = []
        for name, buffer in [('stdout', stdout), ('stderr', stderr)]:
            if buffer.getvalue():
                streams.append(nbformat.v4.new_output('stream', name=name, text=buffer.getvalue()))
        cell.outputs = streams + outputs
        cell.execution_count = count
    nbformat.validate(notebook)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    if update:
        # Keep committed notebooks small; charts live alongside the CSV evidence.
        compact = copy.deepcopy(notebook)
        compact.cells = [c for c in compact.cells if not c.metadata.get('saved_chart')]
        cells = []
        for cell in compact.cells:
            cells.append(cell)
            if cell.cell_type != 'code':
                continue
            cell.outputs = [o for o in cell.outputs if 'image/png' not in o.get('data', {})]
            match = re.search(r'plt.savefig\(OUTPUT_DIR / "([^"]+)"', cell.source)
            if match:
                cells.append(nbformat.v4.new_markdown_cell(
                    f'![Saved figure](results/{match.group(1)})',
                    metadata={'saved_chart': True}))
        compact.cells = cells
        nbformat.write(compact, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', choices=['all', *PROJECTS], default='all')
    parser.add_argument('--update-notebooks', action='store_true', help='Refresh committed notebook outputs too.')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    selected = PROJECTS if args.project == 'all' else {args.project: PROJECTS[args.project]}
    for key, (folder, filename) in selected.items():
        source = ROOT / folder / filename
        output = ROOT / folder / 'results' / 'executed.ipynb'
        if args.worker:
            execute_notebook(source, output, args.update_notebooks)
            return
        command = [sys.executable, str(Path(__file__).resolve()), '--project', key, '--worker']
        if args.update_notebooks:
            command.append('--update-notebooks')
        print(f'Executing {key} in a fresh Python process...', flush=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        subprocess.run(command, cwd=source.parent, env=env, check=True, timeout=600)
        metrics = pd.read_csv(ROOT / folder / 'results' / 'metrics.csv')
        if key == 'iris':
            assert np.allclose(metrics['Accuracy'], [28 / 30, 29 / 30], atol=1e-8)
            assert np.allclose(metrics['F1-score'], [0.9333333333333333, 0.9665831244778613], atol=1e-8)
            repeated = pd.read_csv(ROOT / folder / 'results' / 'repeated_validation_folds.csv')
            assert repeated.groupby('Model').size().eq(50).all()
            assert repeated[['Accuracy', 'Macro F1']].apply(lambda col: col.between(0, 1)).all().all()
        else:
            assert np.allclose(metrics['RMSE ($)'], [68078.326, 67358.059, 67358.064, 67358.062], atol=0.10)
            assert np.allclose(metrics['R²'], [0.664, 0.671, 0.671, 0.671], atol=0.001)
            bands = pd.read_csv(ROOT / folder / 'results' / 'errors_by_price_band.csv')
            assert (bands.groupby('Model')['Areas'].sum() == 3400).all()
            geographic = pd.read_csv(ROOT / folder / 'results' / 'geographic_validation_folds.csv')
            assert geographic.groupby(['Validation', 'Model']).size().eq(5).all()
            assert np.isfinite(geographic[['RMSE ($)', 'MAE ($)']].to_numpy()).all()
        print(f'{key}: all cells executed; historical-result checks passed. Outputs: {output.parent}', flush=True)


if __name__ == '__main__':
    main()
