today=`date -u "+%Y-%m-%d"`
cd daily_arxiv
scrapy crawl arxiv -o ../data/${today}.jsonl

cd ../ai
# Use concurrent processing if CONCURRENT_MODE is set to "true"
if [ "${CONCURRENT_MODE}" = "true" ]; then
    echo "Using concurrent AI enhancement processing..."
    python enhance.py --data ../data/${today}.jsonl --concurrent --rpm-limit ${RPM_LIMIT:-6}
else
    echo "Using traditional serial AI enhancement processing..."
    python enhance.py --data ../data/${today}.jsonl
fi

cd ../to_md
python convert.py --data ../data/${today}_AI_enhanced_${LANGUAGE}.jsonl

cd ..
python update_readme.py
