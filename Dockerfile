FROM golang:1.22.10-bookworm AS build
WORKDIR /src
COPY go.mod go.sum go.work ./
COPY pianopir ./pianopir
COPY pirclient ./pirclient
COPY rpc ./rpc
COPY internal ./internal
COPY anchorann ./anchorann
RUN go build ./... && go test ./... && go vet ./... \
    && mkdir /bin/anchorann \
    && for cmd in cluster build-bundle search-server search-client; do \
         go build -o /bin/anchorann/$cmd ./anchorann/cmd/$cmd; done

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates liblz4-1 \
    && apt-get clean
WORKDIR /artifact
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=build /bin/anchorann /usr/local/bin
COPY . .
ENV MPLBACKEND=Agg
CMD ["python3", "scripts/benchmark.py", "--help"]
