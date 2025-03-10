FROM python:3.12 AS devel

RUN apt-get update && apt-get install --no-install-recommends -y sudo

ARG USERNAME=a1
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd -g $USER_GID $USERNAME

RUN useradd -rm -d /home/$USERNAME -s /bin/bash -g $USER_GID -G sudo -u $USER_UID $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

WORKDIR /home/$USERNAME

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.12 AS prod

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "app/app.py"]
