

conda create -n litreading python=3.9
conda activate litreading

conda env config vars set PYTHONPATH="$(pwd)/prev:$PYTHONPATH"
pip install -r requirements_dev.txt

python -m ipykernel install --user --name=litreading
