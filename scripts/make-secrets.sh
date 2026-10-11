#!/usr/bin/env bash
# Create the certificates and passwords of a secured deployment under
# deploy/secrets/ (git-ignored). Existing passwords and the CA are kept; the
# server certificate is issued again for the names given.
#
# Usage: scripts/make-secrets.sh [HOST_OR_IP...]
#   HOST_OR_IP  names clients will use to reach the board, e.g. its address;
#               localhost and the compose service names are always included
# Result:
#   deploy/secrets/ca/ca.key        signs certificates; never leaves this PC
#   deploy/secrets/run/             what the board needs:
#     ca.crt server.crt server.key  TLS for the dashboard and the MQTT broker
#     http.password                 dashboard and /metrics, user "admin"
#     mqtt.password                 MQTT, user "va"
#     grafana.password              Grafana, user "admin"
#     mosquitto.passwd app.env grafana.env   derived from the above
set -euo pipefail

cd "$(dirname "$0")/../deploy"
umask 077
mkdir -p secrets/ca secrets/run
RUN=secrets/run

# Signs a request with the given key and certificate, valid from yesterday:
# a certificate that starts "now" is rejected by a board whose clock runs a
# little behind this PC's. `openssl x509 -req` cannot set the start date.
#   sign REQUEST OUT DAYS EXTENSIONS SIGNER_KEY [SIGNER_CERT]
sign() {
  local work
  work=$(mktemp -d)
  touch "$work/index.txt"
  openssl rand -hex 8 >"$work/serial"
  cat >"$work/ca.cnf" <<CNF
[ca]
default_ca = this
[this]
new_certs_dir = $work
database = $work/index.txt
serial = $work/serial
default_md = sha256
policy = any
unique_subject = no
[any]
commonName = supplied
[ext]
$4
CNF
  local signer=(-keyfile "$5" -cert "${6:-}")
  [ -n "${6:-}" ] || signer=(-keyfile "$5" -selfsign)
  openssl ca -batch -notext -config "$work/ca.cnf" -extensions ext \
    "${signer[@]}" -in "$1" -out "$2" \
    -startdate "$(date -u -d yesterday +%y%m%d%H%M%SZ)" \
    -enddate "$(date -u -d "+$3 days" +%y%m%d%H%M%SZ)" 2>/dev/null
  rm -rf "$work"
}

if [ ! -f secrets/ca/ca.key ]; then
  openssl req -newkey rsa:2048 -nodes -subj "/CN=video-analytics CA" \
    -keyout secrets/ca/ca.key -out secrets/ca/ca.csr 2>/dev/null
  sign secrets/ca/ca.csr $RUN/ca.crt 3650 \
    "basicConstraints = critical, CA:TRUE" secrets/ca/ca.key
  rm -f secrets/ca/ca.csr
fi

# The certificate must name every way a client addresses the server
SAN="DNS:localhost,IP:127.0.0.1,DNS:app,DNS:mosquitto"
for name in "$@"; do
  if [[ "$name" =~ ^[0-9.]+$ ]]; then SAN="$SAN,IP:$name"; else SAN="$SAN,DNS:$name"; fi
done
openssl req -newkey rsa:2048 -nodes -subj "/CN=video-analytics" \
  -keyout $RUN/server.key -out $RUN/server.csr 2>/dev/null
sign $RUN/server.csr $RUN/server.crt 825 "subjectAltName = $SAN" \
  secrets/ca/ca.key $RUN/ca.crt
rm -f $RUN/server.csr $RUN/ca.srl

for name in http mqtt grafana; do
  # No trailing newline: the files are also read verbatim as passwords
  [ -f $RUN/$name.password ] || openssl rand -hex 16 | tr -d '\n' >$RUN/$name.password
done

# Mosquitto wants its own hashed password file
rm -f $RUN/mosquitto.passwd
if command -v mosquitto_passwd >/dev/null; then
  mosquitto_passwd -b -c $RUN/mosquitto.passwd va "$(cat $RUN/mqtt.password)" 2>/dev/null
else
  docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/$RUN:/out" eclipse-mosquitto:2.0.20 \
    mosquitto_passwd -b -c /out/mosquitto.passwd va "$(cat $RUN/mqtt.password)"
fi

cat >$RUN/app.env <<ENV
VA_HTTP_USER=admin
VA_HTTP_PASSWORD=$(cat $RUN/http.password)
VA_TLS_CERT=/app/secrets/server.crt
VA_TLS_KEY=/app/secrets/server.key
VA_MQTT_PORT=8883
VA_MQTT_USER=va
VA_MQTT_PASSWORD=$(cat $RUN/mqtt.password)
VA_MQTT_CA=/app/secrets/ca.crt
ENV
cat >$RUN/grafana.env <<ENV
GF_SECURITY_ADMIN_PASSWORD=$(cat $RUN/grafana.password)
ENV

echo "Secrets in deploy/secrets/run (certificate valid for: $SAN)"
