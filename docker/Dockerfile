#docker build -t rpy_notebook .
#docker run -p 8888:8888 -v "$PWD":/home/jovyan/work rpy_notebook
FROM jupyter/datascience-notebook

USER root

# R pre-requisites, can most likely delete this
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fonts-dejavu \
    gfortran \
    gcc && \
    rm -rf /var/lib/apt/lists/*

#add channels to install scanpy, gprofiler-official
RUN  conda config --add channels defaults \
    && \
    conda config --add channels bioconda \
    && \
    conda config --add channels conda-forge



RUN apt-get update && apt-get install --yes\
       r-base-dev \
       libssl-dev \
       libcurl4-openssl-dev \
       libxml2-dev \
       libpng-dev; \
    R -e "install.packages(c('class', 'Seurat'), repos = 'https://urldefense.proofpoint.com/v2/url?u=http-3A__lib.stat.cmu.edu_R_CRAN_&d=DwIGAg&c=shNJtf5dKgNcPZ6Yh64b-A&r=IRDkiz1FUo8bq20_WGVGusufh90RFvNCmCrPdmmYjWY&m=NjDCQaydjvDlvGD7_D-oNeqwpAKSiga02-NrHXifN4I&s=WFCLkq73yrkgJyFRk81aiNeJsfETApuQKAZCILzy7UA&e= ')"; \
    apt-get clean;
USER $NB_UID
#install extra python packages
RUN conda install --quiet --yes \
            'louvain=0.6*' \
            'leidenalg=0.7*'\
            'scanpy=1.4.6' \
            'python-igraph=0.7*' \
            'gprofiler-official=1.0.0' \
            'anndata=0.7*' \
            'anndata2ri=1.0*' \
            && \
            conda clean --all -f -y && \
            fix-permissions $CONDA_DIR && \
            fix-permissions /home/$NB_USER
