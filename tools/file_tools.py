import os
import re
import pandas as pd
from  typing import List
from tool_registry import  register_tool
from difflib import get_close_matches
from rapidfuzz import process, fuzz

################ Helpers

def resolve_path(path: str = "") -> str:
    """Resolve a directory path, defaulting to current directory, and validate its existence."""
    # If path is empty or ".", use current directory
    if not path or path == ".":
        return os.getcwd()
    
    # If path is not absolute, make it relative to current directory
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Path not found: '{path}'. Please provide a valid directory.")
    return path


def normalize_center_keyword(center_keyword: str, files: list[str]) -> str:
    # Extract all unique base names from CSV filenames (e.g., neyshabour_maryam from neyshabour_maryam.csv)
    candidates = {os.path.splitext(f)[0].lower() for f in files if f.endswith(".csv")}
    matches = get_close_matches(center_keyword.lower(), candidates, n=1, cutoff=0.6)
    return matches[0] if matches else center_keyword.lower()



################ Constants

EXPECTED_COLUMNS = [
    "نام مرکز",
    "نوع دوره",
    "نام دقیق دوره",
    "تاریخ شروع دوره",
    "تاریخ پایان دوره",
    "نام مددجویان شرکت کننده",
    "نام معلم",
    "تعداد مددجویان شرکت کننده",
    "تعداد جلسات دوره",
    "هزینه پرداختی به معلم",
    "هزینه های اضافی",
    "توضیحات",
]

SYNONYM_MAP = {
    "مبلغ پرداخت شده به معلم بعد از تخفیفات": "هزینه پرداختی به معلم",
    "کل مبلغ واریزی نهایی به مدرس بابت کلاس بعد از تخفیفات. لطفا مبلغ را به تومان وارد کنید": "هزینه پرداختی به معلم",
    "کل مبلغ واریزی به حساب مدرس بعد از تحقیقات (مبلغی که از محل حساب گلستان پرداخت می" : "هزینه پرداختی به معلم",
    "نام و نام خانوادگی معلم": "نام معلم",
    "نام مدرس": "نام معلم",
    "نام و نام خانوادگی مدرس" : "نام معلم",
    "تعداد دانش آموزان": "تعداد مددجویان شرکت کننده",
    "نوع دوره پیشنهادی": "نوع دوره",
    "توضیحات مرتبط-هر نکته ای که در مورد این دوره لازم به ذکر هست بنویسید." : "توضیحات",
    # Add more as needed...
}

COLUMN_TYPES = {
    "text": [
        "نام مرکز", "نوع دوره", "نام دقیق دوره",
        "نام مددجویان شرکت کننده", "نام معلم"
    ],
    "numeric": [
        "تعداد مددجویان شرکت کننده", "تعداد جلسات دوره",
        "هزینه پرداختی به معلم", "هزینه های اضافی"
    ],
    "date": [
        "تاریخ شروع دوره", "تاریخ پایان دوره"
    ]
}


@register_tool(tags=["file_operations", "match"])
def match_columns_in_csv(file_path: str, cutoff: float = 70) -> str:
    """
    Fuzzy-matches expected Farsi column names in the header of a CSV file using synonyms and token-based matching.
    """
    try:
        resolved_file_path = resolve_path(file_path) if not os.path.isfile(file_path) else file_path
        df = pd.read_csv(resolved_file_path)

        if df.empty or df.columns.empty:
            return "❌ File does not have a valid header row."

        actual_columns = [col.strip() for col in df.columns]
        lines = []

        for expected in EXPECTED_COLUMNS:
            synonym_match = next(
                (actual for actual in actual_columns if SYNONYM_MAP.get(actual) == expected),
                None
            )
            if synonym_match:
                lines.append(f"{expected} → {synonym_match} (via synonym)")
                continue

            if expected in actual_columns:
                lines.append(f"{expected} → {expected} (exact match)")
                continue

            match, score, _ = process.extractOne(expected, actual_columns, scorer=fuzz.token_set_ratio)
            if score >= cutoff:
                lines.append(f"{expected} → {match} ({score:.1f})")
            else:
                lines.append(f"{expected} → ❌ Not found")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Error reading file: {e}"

    

@register_tool(tags=["file_operations", "read"])
def read_project_file(name: str) -> str:
    """
    USAGE: Use when user asks to 'read [filename]', 'show [filename]', or 'analyze [filename]'.
    
    Reads the content of a specified file. If the file is a CSV, it performs a fuzzy match on column names instead.
    """
    if not os.path.exists(name):
        return f"❌ File not found: {name}"

    if name.lower().endswith(".csv"):
        return match_columns_in_csv(name)

    try:
        with open(name, "r") as f:
            return f.read()
    except Exception as e:
        return f"❌ Error reading file: {e}"


@register_tool(tags=["file_operations", "list"])
def list_csv_files() -> List[str]:
    """
    USAGE: Use when user asks 'list csv files', 'show csvs', or 'what csvs are here' while already in the target directory.
    
    Lists all CSV files in the current working directory only.
    """
    return sorted([file for file in os.listdir(".") if file.lower().endswith(".csv")])  


import re

@register_tool(tags=["file_operations", "count"])
def count_csv_files(path: str = "") -> str:
    """
    USAGE: Use when user asks 'count csv files', 'how many csv files', 'how many files are here/there'.
    Can detect directory from user query or use current directory.
    
    Counts all CSV files in the specified directory or current directory. 
    Can extract directory names from user queries automatically.
    """
    try:
        # If no path provided, use current directory
        target_path = path or "."
        
        # Try to resolve the path
        if target_path == ".":
            resolved_path = os.getcwd()
            location_desc = "current directory"
        else:
            resolved_path = resolve_path(target_path)
            location_desc = f"directory '{target_path}'"
        
        # Get CSV files
        list_of_csv = sorted([file for file in os.listdir(resolved_path) if file.lower().endswith(".csv")])
        count = len(list_of_csv)
        
        # Format response
        if count == 0:
            return f"No CSV files found in {location_desc}."
        elif count == 1:
            return f"Found 1 CSV file in {location_desc}: {list_of_csv[0]}"
        else:
            files_str = ", ".join(list_of_csv)
            return f"Found {count} CSV files in {location_desc}: {files_str}"
            
    except FileNotFoundError:
        # If path doesn't exist, suggest alternatives
        return f"❌ Directory '{target_path}' not found. Please check the path or use 'change_directory' to navigate to the correct location first."
    except Exception as e:
        return f"❌ Error counting CSV files: {e}"


@register_tool(tags=["file_operations", "count"])  
def smart_count_csv_files(query_context: str = "") -> str:
    """
    USAGE: Use when user asks 'how many csv files in [directory]' or mentions a specific directory in their query.
    
    Intelligently extracts directory path from user query and counts CSV files.
    Handles queries like 'how many files in input_csvs' or 'count csvs in data folder'.
    """
    try:
        # Extract potential directory names from the query context
        directory_patterns = [
            r'\bin\s+([a-zA-Z0-9_\-\/\.]+)',  # "in directory_name"
            r'\bfrom\s+([a-zA-Z0-9_\-\/\.]+)',  # "from directory_name"  
            r'([a-zA-Z0-9_\-]+_[a-zA-Z0-9_\-]+)',  # underscore patterns like "input_csvs"
            r'\b([a-zA-Z0-9_\-]{3,})\b'  # general directory-like words
        ]
        
        detected_path = None
        for pattern in directory_patterns:
            matches = re.findall(pattern, query_context, re.IGNORECASE)
            for match in matches:
                # Check if this looks like a directory
                potential_path = match.strip()
                if os.path.exists(potential_path):
                    detected_path = potential_path
                    break
            if detected_path:
                break
        
        if detected_path:
            # Use the detected path
            list_of_csv = sorted([file for file in os.listdir(detected_path) if file.lower().endswith(".csv")])
            count = len(list_of_csv)
            
            if count == 0:
                return f"No CSV files found in '{detected_path}'."
            elif count == 1:
                return f"Found 1 CSV file in '{detected_path}': {list_of_csv[0]}"
            else:
                files_str = ", ".join(list_of_csv)
                return f"Found {count} CSV files in '{detected_path}': {files_str}"
        else:
            # No path detected, use current directory
            list_of_csv = sorted([file for file in os.listdir(".") if file.lower().endswith(".csv")])
            count = len(list_of_csv)
            
            if count == 0:
                return "No CSV files found in current directory. Please specify a directory path or navigate to the correct location."
            else:
                current_dir = os.path.basename(os.getcwd())
                files_str = ", ".join(list_of_csv)
                return f"Found {count} CSV files in current directory ('{current_dir}'): {files_str}"
                
    except Exception as e:
        return f"❌ Error: {e}. Please specify the directory path clearly or use 'change_directory' to navigate first."


@register_tool(tags=["file_operations", "list"])
def list_csv_files_in_dir(path: str = "") -> List[str]:
    """
    USAGE: Use when user asks 'list all files in [directory]', 'show all csvs in [path]', or 'what files are in [folder]'.
    DO NOT use for center-specific queries.
    
    Lists all CSV files in a specified folder (defaults to current folder if none given).
    """
    try:
        resolved = resolve_path(path)
        return sorted([f for f in os.listdir(resolved) if f.lower().endswith(".csv")])
    except FileNotFoundError as e:
        return [str(e)]



@register_tool(tags=["file_operations"])
def change_directory(path: str) -> str:
    """
    USAGE: Use when user asks to 'go to [directory]', 'cd [path]', 'navigate to [folder]', or 'change to [directory]'.
    
    Changes the current working directory.
    """
    try:
        os.chdir(path)
        return f"Changed working directory to {os.getcwd()}"
    except Exception as e:
        return f"Error changing directory: {e}"
    

@register_tool(tags=["file_operations", "list"])
def list_center_csv_files(center_keyword: str, path: str = "") -> List[str]:
    """
    USAGE: Use when user asks 'list files from [center]', 'show [center] files', 'files related to [center]', or '[center] csvs'.
    
    Lists all CSV files related to a specific center in the given folder (fuzzy matched).
    Examples: 'list files from neyshabour', 'show boushehr files', 'sanandaj csvs'
    """
    try:
        resolved = resolve_path(path)
        all_files = os.listdir(resolved)
        normalized = normalize_center_keyword(center_keyword, all_files)
        return sorted([
            f for f in all_files
            if f.lower().endswith(".csv") and normalized in f.lower()
        ])
    except FileNotFoundError as e:
        return [str(e)]


@register_tool(tags=["file_operations", "count"])
def count_center_csv_files(center_keyword: str, path: str = "") -> str:
    """
    USAGE: Use when user asks 'how many files from [center]', 'count files from [center]', 'how many [center] files', or 'number of [center] csvs'.
    
    Counts how many CSV files are related to a center in the specified folder.
    Examples: 'how many files from neyshabour', 'count boushehr files', 'how many sanandaj csvs'
    """
    try:
        resolved = resolve_path(path)
        all_files = os.listdir(resolved)
        normalized = normalize_center_keyword(center_keyword, all_files)
        matching = [f for f in all_files if f.lower().endswith(".csv") and normalized in f.lower()]
        return f"Found {len(matching)} CSV files related to center '{normalized}' in '{resolved}'."
    except FileNotFoundError as e:
        return str(e)


@register_tool(tags=["file_operations", "count"])
def count_center_csv_files_current_dir(center_keyword: str) -> str:
    """
    USAGE: Use when user asks 'how many files from [center]' while already in the target directory.
    
    Counts how many CSV files are related to a center in the current working directory.
    """
    try:
        all_files = os.listdir(".")
        normalized = normalize_center_keyword(center_keyword, all_files)
        matching = [f for f in all_files if f.lower().endswith(".csv") and normalized in f.lower()]
        return f"Found {len(matching)} CSV files related to center '{normalized}' in current directory."
    except Exception as e:
        return f"❌ Error: {e}"


@register_tool(tags=["file_operations", "list"])
def list_center_csv_files_current_dir(center_keyword: str) -> List[str]:
    """
    USAGE: Use when user asks 'list files from [center]' while already in the target directory.
    
    Lists all CSV files related to a specific center in the current working directory (fuzzy matched).
    """
    try:
        all_files = os.listdir(".")
        normalized = normalize_center_keyword(center_keyword, all_files)
        return sorted([
            f for f in all_files
            if f.lower().endswith(".csv") and normalized in f.lower()
        ])
    except Exception as e:
        return [f"❌ Error: {e}"]


def infer_center_name_from_filename(filename: str) -> str:
    """
    Extract the base center name from a filename using patterns and normalization.
    """
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    # Normalize underscores and remove trailing numbers
    name = re.sub(r"(_?\d+)?$", "", base)
    name = re.sub(r"[^a-zA-Z0-9آ-ی_]+", "", name)
    return name.strip("_")


@register_tool(tags=["file_operations", "clean"])
def clean_csv_file(file_path: str) -> str:
    """
    USAGE: Use when user asks to 'clean [filename]', 'process [filename]', or 'standardize [filename]'.
    
    Cleans and standardizes a CSV file by mapping columns to expected format and normalizing data types.
    """
    try:
        resolved_file_path = resolve_path(file_path) if not os.path.isfile(file_path) else file_path
        df = pd.read_csv(resolved_file_path)
        if df.empty or df.columns.empty:
            return "❌ File has no header or is empty."

        actual_columns = [col.strip() for col in df.columns]
        column_map = {}

        for expected in EXPECTED_COLUMNS:
            synonym_match = next((col for col in actual_columns if SYNONYM_MAP.get(col) == expected), None)
            if synonym_match:
                column_map[expected] = synonym_match
                continue

            if expected in actual_columns:
                column_map[expected] = expected
                continue

            match, score, _ = process.extractOne(expected, actual_columns, scorer=fuzz.token_set_ratio)
            if score >= 70:
                column_map[expected] = match

        reverse_map = {v: k for k, v in column_map.items()}
        df.rename(columns=reverse_map, inplace=True)

        df = df[[col for col in EXPECTED_COLUMNS if col in df.columns]]

        center_name = infer_center_name_from_filename(file_path)
        df["نام مرکز"] = center_name

        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                continue

            if col in COLUMN_TYPES["numeric"]:
                df[col] = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce")

            elif col in COLUMN_TYPES["date"]:
                df[col] = df[col].astype(str).str.replace("-", "/").str.strip()
                df[col] = df[col].where(df[col].str.match(r"\d{4}/\d{2}/\d{2}"), None)

            elif col in COLUMN_TYPES["text"]:
                df[col] = df[col].astype(str).str.strip()

        cleaned_dir = os.path.join("cleaned_csvs")
        os.makedirs(cleaned_dir, exist_ok=True)
        cleaned_path = os.path.join(cleaned_dir, os.path.basename(file_path))
        df.to_csv(cleaned_path, index=False)

        preview = df.head(3).to_string(index=False)
        return f"✅ Cleaned file saved as: {cleaned_path}\n\n🔍 Preview:\n{preview}"

    except Exception as e:
        return f"❌ Error while cleaning CSV: {e}"
    

@register_tool(tags=["file_operations", "clean"])
def clean_all_csv_files(path: str = "") -> str:
    """
    USAGE: Use when user asks 'clean all files', 'clean all csvs', 'process all files in [directory]', 'clean everything', or 'batch clean files'.
    
    Cleans and standardizes all CSV files in the specified directory (or current directory if no path provided).
    Uses the existing clean_csv_file function for each file.
    
    Args:
        path: The directory path (defaults to current directory if empty)
    """
    try:
        # Determine target directory
        if not path or path == ".":
            target_directory = os.getcwd()
            location_desc = "current directory"
        else:
            target_directory = resolve_path(path)
            location_desc = f"directory '{path}'"
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(target_directory) if f.lower().endswith(".csv")]
        
        if not csv_files:
            return f"❌ No CSV files found in {location_desc}."
        
        # Track results
        cleaned_files = []
        failed_files = []
        total_files = len(csv_files)
        
        print(f"🧹 Starting to clean {total_files} CSV files in {location_desc}...")
        
        # Clean each CSV file
        for i, csv_file in enumerate(csv_files, 1):
            file_path = os.path.join(target_directory, csv_file)
            print(f"   [{i}/{total_files}] Processing: {csv_file}")
            
            try:
                # Use the existing clean_csv_file function
                result = clean_csv_file(file_path)
                
                # Check if cleaning was successful
                if result.startswith("✅"):
                    cleaned_files.append(csv_file)
                    print(f"   ✅ Successfully cleaned: {csv_file}")
                else:
                    failed_files.append(csv_file)
                    print(f"   ❌ Failed to clean: {csv_file} - {result}")
                    
            except Exception as e:
                failed_files.append(csv_file)
                print(f"   ❌ Error cleaning {csv_file}: {str(e)}")
        
        # Generate summary report
        summary_lines = [
            f"🧹 BATCH CLEANING COMPLETED",
            f"📁 Location: {location_desc}",
            f"📊 Total files processed: {total_files}",
            f"✅ Successfully cleaned: {len(cleaned_files)}",
            f"❌ Failed to clean: {len(failed_files)}"
        ]
        
        if cleaned_files:
            summary_lines.append(f"\n✅ Cleaned files:")
            for file in cleaned_files:
                summary_lines.append(f"   • {file}")
        
        if failed_files:
            summary_lines.append(f"\n❌ Failed files:")
            for file in failed_files:
                summary_lines.append(f"   • {file}")
        
        summary_lines.append(f"\n📂 Cleaned files saved to: ../cleaned_csvs/")
        
        return "\n".join(summary_lines)
        
    except FileNotFoundError:
        return f"❌ Directory '{path}' not found. Please check the path or use 'change_directory' to navigate to the correct location first."
    except Exception as e:
        return f"❌ Error during batch cleaning: {e}"


@register_tool(tags=["file_operations", "clean"])
def clean_all_csv_files_with_preview(path: str = "", max_preview: int = 2) -> str:
    """
    USAGE: Use when user asks 'clean all files and show preview', 'process all csvs with preview', or wants to see sample results.
    
    Cleans all CSV files in directory and shows preview of first few cleaned files.
    
    Args:
        path: The directory path (defaults to current directory if empty)
        max_preview: Maximum number of files to show preview for (default 2)
    """
    try:
        # First, get the basic cleaning results
        basic_result = clean_all_csv_files(path)
        
        if basic_result.startswith("❌"):
            return basic_result
        
        # Determine target directory for preview
        if not path or path == ".":
            target_directory = os.getcwd()
        else:
            target_directory = resolve_path(path)
        
        # Get cleaned files for preview
        csv_files = [f for f in os.listdir(target_directory) if f.lower().endswith(".csv")]
        preview_files = csv_files[:max_preview]
        
        # Add preview section
        preview_lines = [basic_result, "\n" + "="*50, "📋 PREVIEW OF CLEANED FILES:", "="*50]
        
        for i, csv_file in enumerate(preview_files, 1):
            file_path = os.path.join(target_directory, csv_file)
            try:
                # Read a few rows from the original file for preview
                df = pd.read_csv(file_path)
                preview = df.head(2).to_string(index=False, max_cols=5)
                preview_lines.extend([
                    f"\n📄 File {i}: {csv_file}",
                    f"🔍 Sample data (first 2 rows):",
                    preview[:200] + "..." if len(preview) > 200 else preview
                ])
            except Exception as e:
                preview_lines.append(f"\n📄 File {i}: {csv_file} - Preview unavailable: {e}")
        
        if len(csv_files) > max_preview:
            preview_lines.append(f"\n... and {len(csv_files) - max_preview} more files")
        
        return "\n".join(preview_lines)
        
    except Exception as e:
        return f"❌ Error generating preview: {e}"
    

@register_tool(tags=["file_operations", "consolidate"])
def consolidate_cleaned_csv_files(output_filename: str = "consolidated_data.csv", source_path: str = "cleaned_csvs") -> str:
    """
    USAGE: Use when user asks 'consolidate all files', 'merge all csvs', 'combine all cleaned files', 
    'create consolidated file', or 'merge cleaned data'.
    
    Consolidates all cleaned CSV files from the cleaned_csvs directory into a single CSV file.
    Saves the result in a new 'consolidated_csv' directory.
    
    Args:
        output_filename: Name for the consolidated file (default: "consolidated_data.csv")
        source_path: Path to the cleaned CSV files (default: "cleaned_csvs")
    """
    try:
        # Resolve source path
        if not os.path.exists(source_path):
            return f"❌ Source directory '{source_path}' not found. Please run clean_all_csv_files first."
        
        # Get all CSV files from the cleaned directory
        csv_files = [f for f in os.listdir(source_path) if f.lower().endswith(".csv")]
        
        if not csv_files:
            return f"❌ No CSV files found in '{source_path}' directory."
        
        print(f"🔄 Found {len(csv_files)} files to consolidate...")
        
        # Initialize list to store all dataframes
        all_dataframes = []
        processed_files = []
        failed_files = []
        
        # Process each CSV file
        for i, csv_file in enumerate(csv_files, 1):
            file_path = os.path.join(source_path, csv_file)
            print(f"   [{i}/{len(csv_files)}] Processing: {csv_file}")
            
            try:
                df = pd.read_csv(file_path)
                
                # Verify it has the expected columns structure
                if df.empty:
                    print(f"   ⚠️  Skipping empty file: {csv_file}")
                    continue
                
                # Add source file information for tracking
                df['فایل مبدا'] = csv_file
                
                all_dataframes.append(df)
                processed_files.append(csv_file)
                print(f"   ✅ Added {len(df)} rows from: {csv_file}")
                
            except Exception as e:
                failed_files.append(csv_file)
                print(f"   ❌ Error processing {csv_file}: {str(e)}")
        
        if not all_dataframes:
            return f"❌ No valid data found to consolidate. All files failed or were empty."
        
        # Concatenate all dataframes
        print(f"🔗 Consolidating {len(all_dataframes)} dataframes...")
        consolidated_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        # Ensure consistent column order (put source file column at the end)
        source_col = 'فایل مبدا'
        other_cols = [col for col in consolidated_df.columns if col != source_col]
        consolidated_df = consolidated_df[other_cols + [source_col]]
        
        # Create output directory
        output_dir = "consolidated_csv"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save consolidated file
        output_path = os.path.join(output_dir, output_filename)
        consolidated_df.to_csv(output_path, index=False)
        
        # Generate summary
        total_rows = len(consolidated_df)
        unique_centers = consolidated_df['نام مرکز'].nunique() if 'نام مرکز' in consolidated_df.columns else 0
        
        summary_lines = [
            f"✅ CONSOLIDATION COMPLETED",
            f"📂 Output file: {output_path}",
            f"📊 Total records: {total_rows:,}",
            f"🏢 Unique centers: {unique_centers}",
            f"✅ Successfully processed: {len(processed_files)} files",
            f"❌ Failed files: {len(failed_files)}"
        ]
        
        if processed_files:
            summary_lines.append(f"\n✅ Processed files:")
            for file in processed_files:
                summary_lines.append(f"   • {file}")
        
        if failed_files:
            summary_lines.append(f"\n❌ Failed files:")
            for file in failed_files:
                summary_lines.append(f"   • {file}")
        
        # Add preview of consolidated data
        preview = consolidated_df.head(3).to_string(index=False, max_cols=6)
        summary_lines.extend([
            f"\n🔍 Preview of consolidated data:",
            preview[:300] + "..." if len(preview) > 300 else preview
        ])
        
        return "\n".join(summary_lines)
        
    except Exception as e:
        return f"❌ Error during consolidation: {e}"


@register_tool(tags=["file_operations", "consolidate"])
def consolidate_csv_files_from_directory(source_directory: str, output_filename: str = "consolidated_data.csv") -> str:
    """
    USAGE: Use when user asks 'consolidate files from [directory]', 'merge csvs from [path]', 
    'combine files in [folder]', or specifies a custom source directory.
    
    Consolidates all CSV files from a specified directory into a single CSV file.
    
    Args:
        source_directory: Path to the directory containing CSV files to consolidate
        output_filename: Name for the consolidated file (default: "consolidated_data.csv")
    """
    try:
        # Use the main consolidation function with custom source path
        return consolidate_cleaned_csv_files(output_filename, source_directory)
        
    except Exception as e:
        return f"❌ Error during consolidation: {e}"


@register_tool(tags=["file_operations", "consolidate"])
def consolidate_center_csv_files(center_keyword: str, source_path: str = "cleaned_csvs", output_filename: str = None) -> str:
    """
    USAGE: Use when user asks 'consolidate files from [center]', 'merge [center] files', 
    'combine [center] data', or 'consolidate neyshabour files'.
    
    Consolidates CSV files from a specific center into a single CSV file.
    
    Args:
        center_keyword: The center name to filter files (e.g., "neyshabour", "boushehr")
        source_path: Path to the directory containing CSV files (default: "cleaned_csvs")
        output_filename: Name for the consolidated file (default: auto-generated from center name)
    """
    try:
        # Resolve source path
        if not os.path.exists(source_path):
            return f"❌ Source directory '{source_path}' not found."
        
        # Get all CSV files and filter for the specific center
        all_files = os.listdir(source_path)
        normalized_center = normalize_center_keyword(center_keyword, all_files)
        
        center_files = [
            f for f in all_files 
            if f.lower().endswith(".csv") and normalized_center in f.lower()
        ]
        
        if not center_files:
            return f"❌ No CSV files found for center '{normalized_center}' in '{source_path}'."
        
        print(f"🔄 Found {len(center_files)} files for center '{normalized_center}' to consolidate...")
        
        # Process only the center-specific files
        all_dataframes = []
        processed_files = []
        failed_files = []
        
        for i, csv_file in enumerate(center_files, 1):
            file_path = os.path.join(source_path, csv_file)
            print(f"   [{i}/{len(center_files)}] Processing: {csv_file}")
            
            try:
                df = pd.read_csv(file_path)
                
                if df.empty:
                    print(f"   ⚠️  Skipping empty file: {csv_file}")
                    continue
                
                # Add source file information
                df['فایل مبدا'] = csv_file
                
                all_dataframes.append(df)
                processed_files.append(csv_file)
                print(f"   ✅ Added {len(df)} rows from: {csv_file}")
                
            except Exception as e:
                failed_files.append(csv_file)
                print(f"   ❌ Error processing {csv_file}: {str(e)}")
        
        if not all_dataframes:
            return f"❌ No valid data found for center '{normalized_center}'."
        
        # Consolidate the dataframes
        consolidated_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        # Reorder columns
        source_col = 'فایل مبدا'
        other_cols = [col for col in consolidated_df.columns if col != source_col]
        consolidated_df = consolidated_df[other_cols + [source_col]]
        
        # Generate output filename if not provided
        if not output_filename:
            output_filename = f"{normalized_center}_consolidated.csv"
        
        # Create output directory
        output_dir = "consolidated_csv"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save consolidated file
        output_path = os.path.join(output_dir, output_filename)
        consolidated_df.to_csv(output_path, index=False)
        
        # Generate summary
        total_rows = len(consolidated_df)
        
        summary_lines = [
            f"✅ CENTER CONSOLIDATION COMPLETED",
            f"🏢 Center: {normalized_center}",
            f"📂 Output file: {output_path}",
            f"📊 Total records: {total_rows:,}",
            f"✅ Successfully processed: {len(processed_files)} files",
            f"❌ Failed files: {len(failed_files)}"
        ]
        
        if processed_files:
            summary_lines.append(f"\n✅ Processed files:")
            for file in processed_files:
                summary_lines.append(f"   • {file}")
        
        if failed_files:
            summary_lines.append(f"\n❌ Failed files:")  
            for file in failed_files:
                summary_lines.append(f"   • {file}")
        
        # Add preview
        preview = consolidated_df.head(2).to_string(index=False, max_cols=5)
        summary_lines.extend([
            f"\n🔍 Preview of consolidated data:",
            preview[:250] + "..." if len(preview) > 250 else preview
        ])
        
        return "\n".join(summary_lines)
        
    except Exception as e:
        return f"❌ Error during center consolidation: {e}"