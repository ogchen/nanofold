#!/bin/bash

BASE_URL=https://files.rcsb.org/download
DOWNLOAD_DIR=$1
URLS_FILE=$DOWNLOAD_DIR/urls.txt
DATE=$2
START=0
ROWS=10000

if [[ -z "$DOWNLOAD_DIR" || -z "$DATE" ]]; then
    echo "Usage: $0 <download directory> <date>"
    exit 1
fi

> $URLS_FILE

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
        exit1
    fi

    TOTAL_COUNT=$(echo $CURL_RESULT | grep -o '"total_count" : [0-9]*' | awk '{print $3}')
    if ! [[ $TOTAL_COUNT =~ ^[0-9]+$ ]]; then
        echo "Failed to extract total_count from response, result=$CURL_RESULT. Exiting."
        exit 1
    fi
    
    IDENTIFIERS=$(echo "$CURL_RESULT" | grep -o '"identifier" : "[^"]*' | cut -d '"' -f 4)
    if [[ -z "$IDENTIFIERS" ]]; then
        if [[ $START -lt $TOTAL_COUNT ]]; then
            echo "No identifiers found, result=$CURL_RESULT. Exiting."
            exit 1
        else
            break
        fi
    fi

    for ID in $IDENTIFIERS; do
        echo "$BASE_URL/$ID.cif.gz" >> "$URLS_FILE"
    done
    START=$(expr $START + $ROWS)
done
echo "Fetched URLS to $URL_FILE"

aria2c -d "$DOWNLOAD_DIR" --allow-overwrite=true -i "$URLS_FILE"
