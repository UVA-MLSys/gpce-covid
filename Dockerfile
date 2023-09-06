FROM python:3.9
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get -y update
RUN python -m pip install pip>=23.*
RUN pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pytorch-lightning==1.8.6 \
        pytorch-forecasting==0.10.3 \
        tensorflow==2.10.* \
        ipykernel>=6.20.2 \
        ipywidgets>=7.6.5 \
        jupyter_client>=7.3.5 \
        notebook>=6.5.2 \
        numpy<1.24 \
        setuptools>=67.8