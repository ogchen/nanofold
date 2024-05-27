#!/bin/bash
set -euo pipefail

BASE_URL=https://files.rcsb.org/download
DOWNLOAD_DIR=$1
URLS_FILE=$(mktemp)
DATE=$2
START=0
ROWS=10000

if [[ -z "$DOWNLOAD_DIR" || -z "$DATE" ]]; then
    echo "Usage: $0 <download directory> <date>"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "jq could not be found. Please install jq."
    exit 1
fi

if ! command -v aria2c &> /dev/null; then
    echo "aria2c could not be found. Please install aria2c."
    exit 1
fi

cleanup() {
    if [[ -n "${URLS_FILE}" && -f "$URLS_FILE" ]]; then
        rm -f "$URLS_FILE"
    fi
}

trap cleanup EXIT

while true; do
    QUERY=$(cat <<EOF

{
  "query": {
    "type": "terminal",
    "label": "text",
    "service": "text",
    "parameters": {
      "attribute": "rcsb_accession_info.deposit_date",
      "operator": "less_or_equal",
      "negation": false,
      "value": "$DATE"
    }
  },
  "return_type": "entry",
  "request_options": {
    "paginate": {
      "start": $START,
      "rows": $ROWS
    },
    "results_content_type": [
      "experimental"
    ],
    "sort": [
      {
        "sort_by": "score",
        "direction": "desc"
      }
    ],
    "scoring_strategy": "combined"
  }
}
EOF
    )
    
    CURL_RESULT=$(curl --get --data-urlencode "json=$QUERY" https://search.rcsb.org/rcsbsearch/v2/query)
    CURL_STATUS=$?
    
    if [ $CURL_STATUS -ne 0 ]; then
        echo "curl request failed with status $CURL_STATUS. Exiting."
        exit 1
    fi

    TOTAL_COUNT=$(echo "$CURL_RESULT" | jq '.["total_count"]')
    if ! [[ $TOTAL_COUNT =~ ^[0-9]+$ ]]; then
        echo "Failed to extract total_count from response. Exiting."
        exit 1
    fi
    
    IDENTIFIERS=$(echo "$CURL_RESULT" | jq '.["result_set"][].identifier')
    if [[ -z "$IDENTIFIERS" ]]; then
        if [[ $START -lt $TOTAL_COUNT ]]; then
            echo "No identifiers found, result=$CURL_RESULT. Exiting."
            exit 1
        else
            break
        fi
    fi

    for ID in $IDENTIFIERS; do
        ID=$(echo "$ID" | tr -d '"')
        echo "$BASE_URL/$ID.cif.gz" >> "$URLS_FILE"
    done
    START=$((START + ROWS))
done
echo "Fetched URLS to $URLS_FILE"

aria2c -d "$DOWNLOAD_DIR" --auto-file-renaming=false -i "$URLS_FILE"

if [ ! -d "$DOWNLOAD_DIR" ]; then
  echo "Could not find $DOWNLOAD_DIR. Exiting. Please check if the download was successful."
  exit 1
fi

gzip -d "$DOWNLOAD_DIR"/*.cif.gz
