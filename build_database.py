import os
import glob
import json
import re
from collections import defaultdict
from datetime import datetime

# 定义一个简单的英文停用词列表，用于构建搜索索引时忽略这些常见词
STOP_WORDS = set([
    "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "with",
    "is", "are", "was", "were", "it", "its", "i", "you", "he", "she", "we", "they",
    "as", "at", "by", "from", "that", "this", "which", "who", "what", "where",
    "when", "why", "how", "not", "no", "but", "if", "so", "then", "just", "very"
])

def build_database_from_jsonl():
    """
    构建数据库的主函数。
    它直接从 'data' 目录下的 *_AI_enhanced_Chinese.jsonl 文件中读取结构化数据，然后生成：
    1. 按月份分片的数据文件 (database-YYYY-MM.json)
    2. 一个清单文件 (index.json)
    3. 一个专用的搜索索引文件 (search_index.json)
    4. 一个全新的分类索引文件 (category_index.json)
    """
    monthly_data = defaultdict(list)
    search_index = defaultdict(set)
    category_index = defaultdict(set) # 新增：初始化分类索引
    total_paper_count = 0
    skipped_paper_count = 0

    output_dir = "docs/data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    jsonl_files = glob.glob("data/*_AI_enhanced_Chinese.jsonl")
    if not jsonl_files:
        print("错误: 在 'data' 目录下没有找到任何 '_AI_enhanced_Chinese.jsonl' 文件。")
        return

    print(f"找到 {len(jsonl_files)} 个 .jsonl 数据源文件。开始处理...")

    for jsonl_file in jsonl_files:
        base_name = os.path.basename(jsonl_file)
        date_from_filename_match = re.match(r'(\d{4}-\d{2}-\d{2})', base_name)
        if not date_from_filename_match:
            print(f"警告: 无法从文件名 {base_name} 中提取日期，已跳过。")
            continue
        file_date = date_from_filename_match.group(1)

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    raw_data = json.loads(line)

                    # 核心验证逻辑：只要求论文有ID
                    paper_id = raw_data.get("id")
                    if not paper_id:
                        skipped_paper_count += 1
                        continue

                    ai_enhanced_info = raw_data.get("AI", {})
                    
                    # 关键修正：正确处理keywords字段，将其从字符串分割为数组
                    keywords_str = ai_enhanced_info.get("keywords", "")
                    keywords_list = []
                    if keywords_str and isinstance(keywords_str, str):
                        # 分割字符串并去除每个关键词两端的空格
                        keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

                    # --- 全面、安全地获取所有数据 ---
                    paper_data = {
                        "id": paper_id,
                        "title": raw_data.get("title", "无标题"),
                        "date": file_date, # 使用文件日期作为默认值
                        #"date": raw_data.get("date", file_date), # 使用创建日期或文件日期作为默认值
                        "url": raw_data.get("abs", "#"),
                        "pdf_url": raw_data.get("pdf", "#"), # 直接获取PDF链接
                        "authors": ", ".join(raw_data.get("authors", [])),
                        "abstract": raw_data.get("summary", ""),
                        "comment": raw_data.get("comment", ""),
                        "categories": raw_data.get("categories", []),
                        "updated": raw_data.get("updated", file_date), # 使用更新日期或文件日期作为默认值
                        
                        # 从AI对象中提取所有丰富信息，并修正字段名
                        "zh_title": ai_enhanced_info.get("title_translation"),
                        "translation": ai_enhanced_info.get("translation"), # 摘要翻译
                        "keywords": keywords_list, # 使用处理后的数组
                        "tldr": ai_enhanced_info.get("tldr"),
                        "comments": ai_enhanced_info.get("comments"), # AI点评
                        "motivation": ai_enhanced_info.get("motivation"),
                        "method": ai_enhanced_info.get("method"),
                        "results": ai_enhanced_info.get("result"), # 研究结果
                        "conclusion": ai_enhanced_info.get("conclusion") # 兼容result字段
                    }
                    
                    year_month = file_date[:7]
                    monthly_data[year_month].append(paper_data)
                    total_paper_count += 1

                    # --- 构建搜索索引 ---
                    text_to_index = (paper_data.get("title", "") + " " + paper_data.get("abstract", "")).lower()
                    tokens = re.findall(r'\b[a-z]{3,}\b', text_to_index)
                    for token in tokens:
                        if token not in STOP_WORDS:
                            search_index[token].add(paper_id)
                    
                    if paper_data.get("keywords"):
                        for keyword in paper_data.get("keywords", []):
                            if keyword:
                                search_index[keyword.lower()].add(paper_id)

                    # 新增：构建分类索引
                    if paper_data.get("categories"):
                        for category in paper_data.get("categories", []):
                            category_index[category].add(paper_id)

                except (json.JSONDecodeError, AttributeError):
                    skipped_paper_count += 1
                    continue

    if total_paper_count > 0:
        print(f"处理完成 {total_paper_count} 篇论文。")
        if skipped_paper_count > 0:
            print(f"因缺少ID或格式错误，跳过了 {skipped_paper_count} 篇论文。")
    else:
        print("警告: 未能成功处理任何论文。")
        return

    # --- 开始写入文件 ---
    for month, papers in monthly_data.items():
        sorted_papers = sorted(papers, key=lambda p: p['date'], reverse=True)
        month_file_path = os.path.join(output_dir, f"database-{month}.json")
        with open(month_file_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_papers, f, indent=2, ensure_ascii=False)
    print(f"成功写入 {len(monthly_data)} 个月度数据分片文件。")

    available_months = sorted(monthly_data.keys(), reverse=True)
    # 添加当前日期作为最后更新时间
    current_date = datetime.now().strftime("%Y-%m-%d")
    manifest = {
        "availableMonths": available_months, 
        "totalPaperCount": total_paper_count,
        "lastUpdated": current_date
    }
    manifest_file_path = os.path.join(output_dir, "index.json")
    with open(manifest_file_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("成功写入清单文件 index.json。")

    final_search_index = {token: list(ids) for token, ids in search_index.items()}
    search_index_file_path = os.path.join(output_dir, "search_index.json")
    with open(search_index_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_search_index, f, ensure_ascii=False)
    print("成功写入搜索索引文件 search_index.json。")
    
    # 新增：写入分类索引文件
    final_category_index = {category: list(ids) for category, ids in category_index.items()}
    category_index_file_path = os.path.join(output_dir, "category_index.json")
    with open(category_index_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_category_index, f, ensure_ascii=False)
    print("成功写入分类索引文件 category_index.json。")
    
    old_db_path = "docs/database.json"
    if os.path.exists(old_db_path):
        os.remove(old_db_path)
        print(f"已删除旧的数据文件: {old_db_path}")

if __name__ == "__main__":
    build_database_from_jsonl()