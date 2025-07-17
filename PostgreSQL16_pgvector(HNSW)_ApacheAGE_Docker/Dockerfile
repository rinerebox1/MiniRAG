# syntax=docker/dockerfile:1
########################
# 1. builder stage
########################
ARG PG_MAJOR=16
ARG PGVECTOR_VER=v0.8.0
ARG AGE_VER=PG16/v1.5.0-rc0

FROM postgres:${PG_MAJOR}-bookworm AS builder

# Dockerfile のビルド時変数（ARG）は、その変数を宣言した後の最初の FROM 以降でしか有効になりません。
# そのため、ステージ内で利用する ARG を再宣言しています。
ARG PG_MAJOR
ARG PGVECTOR_VER
ARG AGE_VER

# 必要ライブラリ（ヘッダ込み）をインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
        postgresql-server-dev-${PG_MAJOR} \
        libreadline-dev zlib1g-dev \
        flex bison \
    && rm -rf /var/lib/apt/lists/*

########################
# 1-A. build pgvector
########################
WORKDIR /tmp
RUN git clone --depth 1 --branch ${PGVECTOR_VER} https://github.com/pgvector/pgvector.git \
 && cd pgvector \
 && make && make install

########################
# 1-B. build Apache AGE
########################
WORKDIR /tmp
RUN git clone --depth 1 --branch ${AGE_VER} https://github.com/apache/age.git \
 && cd age \
 && make PG_CONFIG=/usr/bin/pg_config install

########################
# 2. runtime stage
########################
FROM postgres:${PG_MAJOR}-bookworm

# 必要なら再宣言
ARG PG_MAJOR

# ビルド済み .so / .control / SQL をコピー
COPY --from=builder /usr/lib/postgresql/${PG_MAJOR}/lib/*.so /usr/lib/postgresql/${PG_MAJOR}/lib/
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/extension/*vector* /usr/share/postgresql/${PG_MAJOR}/extension/
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/extension/*age*    /usr/share/postgresql/${PG_MAJOR}/extension/

# Apache AGE をプリロード設定
RUN echo "shared_preload_libraries = 'age'" >> /usr/share/postgresql/postgresql.conf.sample
# もしくはこっち？
# RUN echo "shared_preload_libraries = 'age'" >> /usr/share/postgresql/16/postgresql.conf.sample

# 簡易ヘルスチェック（環境変数はランタイムで解決）
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" || exit 1

EXPOSE 5432