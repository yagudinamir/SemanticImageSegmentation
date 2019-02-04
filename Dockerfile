FROM python:3.6


WORKDIR /work_dir
ENV HOME /home
COPY environment.yml ./
RUN apt-get update
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
RUN bash miniconda.sh -b -p $HOME/miniconda
ENV PATH "$HOME/miniconda/bin:$PATH"
RUN hash -r
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN conda info -a
RUN conda env create -f environment.yml
