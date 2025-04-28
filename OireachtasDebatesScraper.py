"""
OireachtasDebatesScraper: Scrape and analyse Dáil debates from the Oireachtas website.

Key Features:
- Scrapes debates for a specified date range.
- Identifies and processes debates mentioning specified search terms.
- Supports Boolean logic (AND/OR/NOT) and regex searches.
- Removes duplicates and cleans text.
- Saves processed text to individual `.txt` files.
- Includes frequency, co-occurrence, and metadata analysis.

Requirements:
- Selenium WebDriver (ChromeDriver).
- Python libraries: pandas, BeautifulSoup, re, tqdm, matplotlib, seaborn, wordcloud, textblob, schedule, requests.

Author: Ruairí Weiner

License: MIT

Citation:
If you use this software, please cite as:
Ruairí Weiner. (2025). OireachtasDebatesScraper: A Python tool for scraping and analysing Irish parliamentary debates.

"""

__version__ = "1.0.0"
__all__ = [
    "DebScraper", "ProcessAndSaveTexts", "ScrapeDebates", "AnalyseFrequencies",
    "VisualiseFrequencies", "GenerateWordCloud", "ExportCombinedTexts",
    "AnalyseTermCooccurrence", "VisualiseCooccurrenceMatrix", "ExportToExcel",
    "ScrapeParliamentarianNames", "CountParliamentarianMentions", "CountParliamentarianDebates",
    "CountPartyMentions", "CountPartyDebates", "CountConstituencyMentions",
    "CountConstituencyDebates", "CreateCaseLevelData", "ScheduleScraping"
]

import argparse
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import logging
import schedule
import requests
import warnings
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from selenium.webdriver.remote.remote_connection import RemoteConnection
import urllib3
import sys
import math
import datetime as dt
from PIL import ImageFont

# Simplified warning and logging configuration - more performant
warnings.filterwarnings("ignore")
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('selenium').setLevel(logging.ERROR)

# Set RemoteConnection.RETRY directly without additional overhead
RemoteConnection.RETRY = False

# Disable retries in urllib3
class NoRetryAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        kwargs["max_retries"] = Retry(total=0)  # Disable retries
        super().__init__(*args, **kwargs)

# Apply the NoRetryAdapter globally to requests
session = requests.Session()
session.mount("http://", NoRetryAdapter())
session.mount("https://", NoRetryAdapter())

def CustomWarning(message):
    """
    Print a custom warning message.

    Args:
        message: The warning message to display.
    """
    print(f"WARNING: {message}")


def SetupWebDriver(chromeDriverPath='/Applications/chromedriver', chromeBinaryPath="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
    """
    Set up and return a Selenium WebDriver instance.

    Args:
        chromeDriverPath (str): Path to the ChromeDriver executable.
        chromeBinaryPath (str): Path to the Google Chrome binary.

    Returns:
        webdriver.Chrome: A Selenium WebDriver instance.

    Raises:
        WebDriverException: If the WebDriver fails to initialise.
    """
    try:
        chromeOptions = Options()
        chromeOptions.binary_location = chromeBinaryPath
        chromeOptions.add_argument("--headless")
        chromeOptions.add_argument("--disable-gpu")
        chromeOptions.add_argument("--no-sandbox")
        chromeOptions.add_argument("--disable-dev-shm-usage")

        service = Service(chromeDriverPath)
        driver = webdriver.Chrome(service=service, options=chromeOptions)
        return driver
    except WebDriverException as e:
        print(f"WebDriver initialization error: {e}")
        exit(1)


def SetupProxy(proxyAddress):
    """
    Set up a proxy for the WebDriver.

    Args:
        proxyAddress: The proxy address (e.g., "http://123.45.67.89:8080").

    Returns:
        A Selenium WebDriver instance with the proxy configured.
    """
    chromeOptions = Options()
    chromeOptions.add_argument(f"--proxy-server={proxyAddress}")
    service = Service('/Applications/chromedriver')
    driver = webdriver.Chrome(service=service, options=chromeOptions)
    logging.info(f"Proxy set up: {proxyAddress}")
    return driver


def ParseBooleanExpression(expression):
    """
    Parse a Boolean expression string into a nested structure.

    Args:
        expression: A string containing a Boolean expression (e.g., "health AND medical OR wellness").

    Returns:
        A nested structure representing the parsed Boolean expression.
    """
    # Replace parentheses for grouping
    expression = expression.replace("(", " ( ").replace(")", " ) ")
    tokens = expression.split()

    def parse(tokens):
        stack = []
        current = []
        for token in tokens:
            if token == "(":
                stack.append(current)
                current = []
            elif token == ")":
                if stack:
                    group = current
                    current = stack.pop()
                    current.append(group)
                else:
                    return current
            elif token.upper() in {"AND", "OR", "NOT"}:
                current.append(token.upper())
            else:
                current.append(token)
        return current

    return parse(tokens)


def EvaluateBooleanExpression(parsedExpression, text):
    """
    Evaluate a parsed Boolean expression against a given text.

    Args:
        parsedExpression: A nested structure representing the parsed Boolean expression.
        text: The text to evaluate the expression against.

    Returns:
        bool: True if the expression matches the text, False otherwise.
    """
    if isinstance(parsedExpression, str):
        # Check if the term is in the text (case-insensitive)
        result = parsedExpression.lower() in text.lower()
        return result

    if isinstance(parsedExpression, list):
        if len(parsedExpression) == 1:
            # Handle single terms like ['renewable']
            return EvaluateBooleanExpression(parsedExpression[0], text)
        if "AND" in parsedExpression:
            # Evaluate all sub-expressions joined by AND
            results = [EvaluateBooleanExpression(subExpr, text) for subExpr in parsedExpression if subExpr != "AND"]
            result = all(results)
            return result
        elif "OR" in parsedExpression:
            # Evaluate any sub-expression joined by OR
            results = [EvaluateBooleanExpression(subExpr, text) for subExpr in parsedExpression if subExpr != "OR"]
            result = any(results)
            return result
        elif "NOT" in parsedExpression:
            # Evaluate NOT (negation of the next sub-expression)
            result = not EvaluateBooleanExpression(parsedExpression[1], text)
            return result

    # Default case: Return False if the structure is unrecognised
    return False


def ValidateDates(startDate, endDate):
    """
    Validate the start and end dates provided by the user.

    Args:
        startDate: Start date as a datetime object.
        endDate: End date as a datetime object.

    Raises:
        ValueError: If the dates are invalid.
    """
    if startDate > endDate:
        raise ValueError("Start date cannot be after the end date.")
    if endDate > datetime.now():
        raise ValueError("End date cannot be in the future.")
    logging.info(f"Validated dates: {startDate} to {endDate}")


def ProcessAndSaveTexts(debatesDf, outputDir, searchTerms="CervicalCheck", regexPattern=None, exportMetadataFormat=None, metadataPath="processed_metadata"):
    """
    Process debates for mentions of specific search terms or regex patterns and save results to text files.

    Args:
        debatesDf: DataFrame containing debates data.
        outputDir: Directory to save the processed text files.
        searchTerms: A single search term (string) or a Boolean expression (e.g., "health AND medical OR wellness").
        regexPattern: Regex pattern to match (default: None).
        exportMetadataFormat: Format to export metadata (e.g., "csv", "json"). Default is None (no export).
        metadataPath: Base path for the exported metadata file (default: "processed_metadata").
    """
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Parse the Boolean expression if provided
    parsedExpression = None
    if searchTerms:
        parsedExpression = ParseBooleanExpression(searchTerms)

    processedTexts = set()  # Track processed texts to avoid duplicates
    metadata = []  # Store metadata for export

    for _, row in debatesDf.iterrows():
        for i in range(1, 50):  # Iterate through potential debate pages
            htmlText = row.get(f'text_{i}')
            if pd.isna(htmlText):  # Skip if the text is missing
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(htmlText, 'html.parser')

            # Extract the page title and body text
            heading = soup.title.get_text() if soup.title else ''
            body = soup.body.get_text() if soup.body else ''

            # Check for matches using regex or Boolean logic
            if regexPattern:
                if not re.search(regexPattern, body, re.IGNORECASE):
                    continue
            elif parsedExpression:
                if not EvaluateBooleanExpression(parsedExpression, body):
                    continue

            # Clean up the text by removing multiple blank lines
            body = re.sub(r'\n\s*\n', '\n', body)

            # Skip if the text is a duplicate
            if body in processedTexts:
                continue

            processedTexts.add(body)

            # Generate a valid filename based on the date and page number
            filename = re.sub(r'[<>:"/\\|?*]', '_', f"{row['date'].strftime('%Y-%m-%d')}_{i}.txt")
            filepath = os.path.join(outputDir, filename)

            # Save the processed text to a file
            with open(filepath, "w", encoding='utf-8') as file:
                file.write(heading + "\n" + body)

            # Add metadata for this file
            metadata.append({"Date": row['date'].strftime('%Y-%m-%d'), "Filename": filename, "Matched Terms": searchTerms})

    # Export metadata if specified
    if exportMetadataFormat:
        metadataDf = pd.DataFrame(metadata)
        if exportMetadataFormat.lower() == "csv":
            metadataDf.to_csv(f"{metadataPath}.csv", index=False)
            print(f"\nMetadata exported to {metadataPath}.csv")
        elif exportMetadataFormat.lower() == "json":
            metadataDf.to_json(f"{metadataPath}.json", orient="records", indent=4)
            print(f"\nMetadata exported to {metadataPath}.json")
        else:
            print(f"\nUnsupported metadata export format: {exportMetadataFormat}. Supported formats are 'csv' and 'json'.")

def ScrapeDayWorker(driver, currentDate, delay, searchTerms, regexPattern):
    """
    Scrape debates for a single day using a provided WebDriver.
    Uses requests/BeautifulSoup to check if a page exists before using Selenium.
    """
    workerData = {'date': currentDate}
    previousText = None
    session = requests.Session()
    for i in range(1, 50):
        url = f"https://www.oireachtas.ie/en/debates/debate/dail/{currentDate.strftime('%Y-%m-%d')}/{i}/"
        # Fast existence check with requests
        try:
            resp = session.head(url, allow_redirects=True, timeout=5)
            # If the HEAD request is redirected to the homepage or another page, treat as not found
            if resp.status_code == 404 or resp.url.rstrip('/') != url.rstrip('/'):
                break  # No more pages for this day
        except Exception:
            # If requests fails, fallback to Selenium (could be network issue)
            pass
        # Now use Selenium to get the page content
        driver.get(url)
        text = driver.page_source
        if text != previousText:
            workerData[f'text_{i}'] = text
        else:
            break
        previousText = text
        if delay and delay > 0:
            time.sleep(delay)
    return workerData

def ScrapeDebates(driver, startDate, endDate, delay=None, searchTerms="CervicalCheck", regexPattern=None, outputDir="ProcessedDebates", exportRawFormat=None, rawPath="raw_scraped_data"):
    """
    Scrape debates for the specified date range using a single WebDriver instance.
    """
    # Determine the delay value
    if delay is True:
        delay = 2  # Default delay of 2 seconds
    elif isinstance(delay, int) and delay > 0:
        delay = delay  # Use the custom delay provided by the user
    else:
        delay = 0  # No delay

    debatesData = []
    currentDate = startDate
    totalDays = (endDate - startDate).days + 1
    scrapedDays = 0

    def formatTime(seconds):
        # Returns a string like HH:MM:SS
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02}:{m:02}:{s:02}"

    def printProgress(scraped, total, elapsed, avgTime):
        barLen = 20
        filledLen = int(round(barLen * scraped / float(total)))
        bar = '█' * filledLen + '-' * (barLen - filledLen)
        percents = 100.0 * scraped / float(total)
        if scraped > 0:
            eta = avgTime * (total - scraped)
        else:
            eta = 0
        sys.stdout.write(
            f"\r[{formatTime(elapsed)}<{formatTime(eta)}, {avgTime:.2f}s/day] |{bar}| {scraped}/{total} [{percents:5.1f}%]"
        )
        sys.stdout.flush()

    startTime = time.time()
    times = []

    try:
        while currentDate <= endDate:
            iterStart = time.time()
            try:
                dateData = ScrapeDayWorker(driver, currentDate, delay, searchTerms, regexPattern)
                debatesData.append(dateData)
                scrapedDays += 1
                iterTime = time.time() - iterStart
                times.append(iterTime)
                avgTime = sum(times) / len(times)
                elapsed = time.time() - startTime
                printProgress(scrapedDays, totalDays, elapsed, avgTime)
            except Exception as e:
                print(f"\nError scraping date {currentDate.strftime('%Y-%m-%d')}: {e}")
                # Optionally, restart driver here if needed
            currentDate += timedelta(days=1)
        print()  # Move to next line after progress bar
    except KeyboardInterrupt:
        print("\nScraping interrupted by user. Exiting gracefully...")
        print(f"Scraped debates for {scrapedDays} days before interruption.")
        if debatesData:
            debatesDf = pd.DataFrame(debatesData)
            debatesDf.to_pickle('DebatesData.pkl')
            print("Saved partial scraped data to 'DebatesData.pkl'.")
        else:
            print("No debates were scraped.")
        return pd.DataFrame(debatesData)

    debatesDf = pd.DataFrame(debatesData)

    # Export raw data if specified
    if exportRawFormat:
        if exportRawFormat.lower() == "csv":
            debatesDf.to_csv(f"{rawPath}.csv", index=False)
            print(f"\nRaw data exported to {rawPath}.csv")
        elif exportRawFormat.lower() == "json":
            debatesDf.to_json(f"{rawPath}.json", orient="records", indent=4)
            print(f"\nRaw data exported to {rawPath}.json")
        else:
            print(f"\nUnsupported raw data export format: {exportRawFormat}. Supported formats are 'csv' and 'json'.")

    print(f"\nSuccessfully scraped debates for {scrapedDays} days. Raw data saved to 'DebatesData.pkl'.")
    debatesDf.to_pickle('DebatesData.pkl')
    return debatesDf

def GenerateSummaryStatistics(debatesDf, processedTexts, exportFormat=None, exportPath="summary_statistics"):
    """
    Generate and display summary statistics for the scraping process.

    Args:
        debatesDf: DataFrame containing the scraped debates data.
        processedTexts: Set of processed texts that matched the search criteria.
        exportFormat: Format to export the results (e.g., "csv", "excel", "json"). Default is None (no export).
        exportPath: Base path for the exported file (default: "summary_statistics").

    Returns:
        A dictionary containing the summary statistics.
    """
    totalDates = len(debatesDf)  # Renamed from "Total Debates Scraped" to "Total Dates Scraped"
    totalPages = sum(len([col for col in row.keys() if col.startswith("text_")]) for _, row in debatesDf.iterrows())
    totalMatches = len(processedTexts)

    summary = {
        "Total Dates Scraped": totalDates,
        "Total Pages Scraped": totalPages,
        "Total Matches Found": totalMatches,
    }

    print("\nSummary Statistics:")
    print("-------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Export the summary if an export format is specified
    if exportFormat:
        summaryDf = pd.DataFrame([summary])  # Convert summary to a DataFrame
        if exportFormat.lower() == "csv":
            summaryDf.to_csv(f"{exportPath}.csv", index=False)
            print(f"\nSummary statistics exported to {exportPath}.csv")
        elif exportFormat.lower() == "excel":
            summaryDf.to_excel(f"{exportPath}.xlsx", index=False)
            print(f"\nSummary statistics exported to {exportPath}.xlsx")
        elif exportFormat.lower() == "json":
            summaryDf.to_json(f"{exportPath}.json", orient="records", indent=4)
            print(f"\nSummary statistics exported to {exportPath}.json")
        else:
            print(f"\nUnsupported export format: {exportFormat}. Supported formats are 'csv', 'excel', and 'json'.")

    return summary

def AnalyseFrequencies(debatesDf, processedTexts, searchTerms, exportFormat=None, exportPath="frequency_analysis"):
    """
    Analyse the frequency of each term and the overall search frequency over time.

    Args:
        debatesDf: DataFrame containing the scraped debates data.
        processedTexts: Set of processed texts that matched the search criteria.
        searchTerms: A single search term (string) or a Boolean expression.
        exportFormat: Format to export the results (e.g., "csv", "excel", "json"). Default is None (no export).
        exportPath: Base path for the exported file (default: "frequency_analysis").

    Returns:
        A pandas DataFrame containing term frequencies and overall frequencies by date.
    """
    # Ensure searchTerms is a list for individual term analysis
    if isinstance(searchTerms, str):
        searchTerms = [searchTerms]

    termFrequencies = {term: {} for term in searchTerms}
    overallFrequency = {}

    for _, row in debatesDf.iterrows():
        date = row['date'].strftime('%Y-%m-%d')
        overallFrequency[date] = 0

        for term in searchTerms:
            termFrequencies[term][date] = 0

        for i in range(1, 50):  # Iterate through potential debate pages
            htmlText = row.get(f'text_{i}')
            if pd.isna(htmlText):  # Skip if the text is missing
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(htmlText, 'html.parser')
            body = soup.body.get_text() if soup.body else ''

            # Count occurrences of each term
            for term in searchTerms:
                termCount = body.lower().count(term.lower())
                termFrequencies[term][date] += termCount

            # Count overall matches (if the page matches any term)
            if any(term.lower() in body.lower() for term in searchTerms):
                overallFrequency[date] += 1

    # Convert term frequencies and overall frequency into a DataFrame
    frequencyData = {"Date": list(overallFrequency.keys()), "Overall Frequency": list(overallFrequency.values())}
    for term, frequencies in termFrequencies.items():
        frequencyData[term] = [frequencies[date] for date in overallFrequency.keys()]

    frequencyDf = pd.DataFrame(frequencyData)

    # Print frequency analysis
    print("\nFrequency Analysis:")
    print("-------------------")
    print(frequencyDf)

    # Export the DataFrame if an export format is specified
    if exportFormat:
        if exportFormat.lower() == "csv":
            frequencyDf.to_csv(f"{exportPath}.csv", index=False)
            print(f"\nFrequency analysis exported to {exportPath}.csv")
        elif exportFormat.lower() == "excel":
            frequencyDf.to_excel(f"{exportPath}.xlsx", index=False)
            print(f"\nFrequency analysis exported to {exportPath}.xlsx")
        elif exportFormat.lower() == "json":
            frequencyDf.to_json(f"{exportPath}.json", orient="records", indent=4)
            print(f"\nFrequency analysis exported to {exportPath}.json")
        else:
            print(f"\nUnsupported export format: {exportFormat}. Supported formats are 'csv', 'excel', and 'json'.")

    return frequencyDf


def VisualiseFrequencies(frequencyDf, searchTerms, outputPath=None):
    """
    Visualise the frequency of each term and the overall search frequency over time.

    Args:
        frequencyDf: DataFrame containing term frequencies and overall frequencies by date.
        searchTerms: A list of search terms to include in the visualisation.
        outputPath: Path to save the visualisation as an image file (default: None, no file saved).

    Returns:
        None
    """
    # Ensure searchTerms is a list
    if isinstance(searchTerms, str):
        searchTerms = [searchTerms]

    # Set the style for the plot
    sns.set(style="whitegrid")

    # Melt the DataFrame for easier plotting with seaborn
    meltedDf = frequencyDf.melt(id_vars=["Date"], 
                                value_vars=["Overall Frequency"] + searchTerms, 
                                var_name="Term", 
                                value_name="Frequency")

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=meltedDf, x="Date", y="Frequency", hue="Term", marker="o")

    # Customise the plot
    plt.title("Frequency of Search Terms Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Search Terms", fontsize=10)
    plt.tightLayout()

    # Save the plot if an output path is specified
    if outputPath:
        plt.savefig(outputPath, dpi=300)
        print(f"Visualisation saved to {outputPath}")

    # Show the plot
    plt.show()


def ExportCombinedTexts(outputDir, combinedFilePath="combined_texts.txt"):
    """
    Combine all processed text files into a single file.

    Args:
        outputDir: Directory containing the processed text files.
        combinedFilePath: Path to save the combined file (default: "combined_texts.txt").
    """
    with open(combinedFilePath, "w", encoding="utf-8") as combinedFile:
        for filename in os.listdir(outputDir):
            if filename.endswith(".txt"):
                with open(os.path.join(outputDir, filename), "r", encoding="utf-8") as file:
                    combinedFile.write(file.read() + "\n")
    print(f"Combined texts saved to {combinedFilePath}")


def AnalyseTermCooccurrence(debatesDf, searchTerms):
    """
    Analyse co-occurrence of search terms in the debates.

    Args:
        debatesDf: DataFrame containing the scraped debates data.
        searchTerms: A list of search terms to analyse.

    Returns:
        A co-occurrence matrix as a pandas DataFrame.
    """
    cooccurrence = {term: {other: 0 for other in searchTerms} for term in searchTerms}

    for _, row in debatesDf.iterrows():
        for i in range(1, 50):
            htmlText = row.get(f"text_{i}")
            if pd.isna(htmlText):
                continue

            # Parse the HTML content
            soup = BeautifulSoup(htmlText, "html.parser")
            body = soup.body.get_text() if soup.body else ""

            # Check co-occurrence of terms
            for term in searchTerms:
                if term.lower() in body.lower():
                    for other in searchTerms:
                        if other.lower() in body.lower():
                            cooccurrence[term][other] += 1

    return pd.DataFrame(cooccurrence)


def VisualiseCooccurrenceMatrix(cooccurrenceMatrix, outputPath=None):
    """
    Visualise the term co-occurrence matrix as a heatmap.

    Args:
        cooccurrenceMatrix: A pandas DataFrame representing the co-occurrence matrix.
        outputPath: Path to save the heatmap image (default: None, no file saved).
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrenceMatrix, annot=True, fmt="d", cmap="coolwarm")
    plt.title("Term Co-occurrence Heatmap", fontsize=16)
    plt.tightLayout()

    if outputPath:
        plt.savefig(outputPath, dpi=300)
        print(f"Heatmap saved to {outputPath}")
        logging.info(f"Heatmap saved to {outputPath}")

    plt.show()


def ExportToExcel(dataframe, outputPath="output.xlsx"):
    """
    Export a DataFrame to an Excel file.

    Args:
        dataframe: The pandas DataFrame to export.
        outputPath: Path to save the Excel file (default: "output.xlsx").
    """
    dataframe.to_excel(outputPath, index=False)
    print(f"Data exported to {outputPath}")
    logging.info(f"Data exported to Excel at {outputPath}")


def ScrapeParliamentarianNames():
    """
    Scrape the names, parties, and constituencies of all TDs and Senators from kildarestreet.com.

    Returns:
        A dictionary with two keys: "TDs" and "Senators", each containing a list of dictionaries with "name", "party", and "constituency".
    """
    baseUrl = "https://www.kildarestreet.com"
    parliamentarians = {"TDs": [], "Senators": []}

    # Scrape TDs
    tdsUrl = f"{baseUrl}/tds/?all=1"
    response = requests.get(tdsUrl)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        for row in soup.select("td.row-1"):
            nameTag = row.find("a")
            if not nameTag:
                continue  # Skip rows without a name
            name = nameTag.get_text()

            partyTag = row.find_next_sibling("td")
            party = partyTag.get_text() if partyTag else "Unknown"

            constituencyTag = partyTag.find_next_sibling("td") if partyTag else None
            constituency = constituencyTag.get_text() if constituencyTag else "Unknown"

            parliamentarians["TDs"].append({"name": name, "party": party, "constituency": constituency})

    # Scrape Senators
    senatorsUrl = f"{baseUrl}/senators/?all=1"
    response = requests.get(senatorsUrl)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        for row in soup.select("td.row-1"):
            nameTag = row.find("a")
            if not nameTag:
                continue  # Skip rows without a name
            name = nameTag.get_text()

            partyTag = row.find_next_sibling("td")
            party = partyTag.get_text() if partyTag else "Unknown"

            constituencyTag = partyTag.find_next_sibling("td") if partyTag else None
            constituency = constituencyTag.get_text() if constituencyTag else "Unknown"

            parliamentarians["Senators"].append({"name": name, "party": party, "constituency": constituency})

    return parliamentarians


def CountParliamentarianMentions(debatesDf, searchTerm):
    """
    Count the total number of mentions of each parliamentarian for a specific search term.

    Args:
        debatesDf: DataFrame containing the scraped debates data.
        searchTerm: The search term to filter debates.

    Returns:
        A pandas DataFrame with parliamentarians and their total mention counts.
    """
    parliamentarians = ScrapeParliamentarianNames()
    counts = {name["name"]: 0 for name in parliamentarians["TDs"] + parliamentarians["Senators"]}

    for _, row in debatesDf.iterrows():
        for i in range(1, 50):  # Iterate through potential debate pages
            htmlText = row.get(f"text_{i}")
            if pd.isna(htmlText):  # Skip if the text is missing
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(htmlText, "html.parser")
            body = soup.body.get_text() if soup.body else ""

            # Check if the search term is in the text
            if searchTerm.lower() in body.lower():
                # Count mentions for each parliamentarian
                for name in counts.keys():
                    counts[name] += body.lower().count(name.lower())

    # Convert counts to a DataFrame
    mentionsDf = pd.DataFrame(list(counts.items()), columns=["Parliamentarian", "Total Mentions"])
    mentionsDf = mentionsDf[mentionsDf["Total Mentions"] > 0]  # Filter out parliamentarians with zero mentions
    return mentionsDf


def CountParliamentarianDebates(debatesDf, searchTerm):
    """
    Count the number of debates in which each parliamentarian is mentioned for a specific search term.

    Args:
        debatesDf: DataFrame containing the scraped debates data.
        searchTerm: The search term to filter debates.

    Returns:
        A pandas DataFrame with parliamentarians and their debate counts.
    """
    parliamentarians = ScrapeParliamentarianNames()
    counts = {name["name"]: 0 for name in parliamentarians["TDs"] + parliamentarians["Senators"]}

    for _, row in debatesDf.iterrows():
        debateMentions = set()
        for i in range(1, 50):  # Iterate through potential debate pages
            htmlText = row.get(f"text_{i}")
            if pd.isna(htmlText):  # Skip if the text is missing
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(htmlText, "html.parser")
            body = soup.body.get_text() if soup.body else ""

            # Check if the search term is in the text
            if searchTerm.lower() in body.lower():
                # Track parliamentarians mentioned in this debate
                for name in counts.keys():
                    if name.lower() in body.lower():
                        debateMentions.add(name)

        # Update debate counts
        for name in debateMentions:
            counts[name] += 1

    # Convert counts to a DataFrame
    debatesDf = pd.DataFrame(list(counts.items()), columns=["Parliamentarian", "Debates Mentioned"])
    debatesDf = debatesDf[debatesDf["Debates Mentioned"] > 0]  # Filter out parliamentarians with zero debates
    return debatesDf


def CountPartyMentions(debatesDf, searchTerm):
    """
    Count the total number of mentions of each party for a specific search term.

    Args:
        debatesDf: DataFrame containing the scraped debates data.
        searchTerm: The search term to filter debates.

    Returns:
        A pandas DataFrame with parties and their total mention counts.
    """
    parliamentarians = ScrapeParliamentarianNames()
    partyCounts = {}

    # Initialise party counts
    for group in ["TDs", "Senators"]:
        for member in parliamentarians[group]:
            party = member["party"]
            if party not in partyCounts:
                partyCounts[party] = 0

    # Count mentions
    for _, row in debatesDf.iterrows():
        for i in range(1, 50):  # Iterate through potential debate pages
            htmlText = row.get(f"text_{i}")
            if pd.isna(htmlText):  # Skip if the text is missing
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(htmlText, "html.parser")
            body = soup.body.get_text() if soup.body else ""

            # Check if the search term is in the text
            if searchTerm.lower() in body.lower():
                # Count mentions for each party
                for group in ["TDs", "Senators"]:
                    for member in parliamentarians[group]:
                        if member["name"].lower() in body.lower():
                            partyCounts[member["party"]] += body.lower().count(member["name"].lower())

    # Convert counts to a DataFrame
    partyMentionsDf = pd.DataFrame(list(partyCounts.items()), columns=["Party", "Total Mentions"])
    partyMentionsDf = partyMentionsDf[partyMentionsDf["Total Mentions"] > 0]  # Filter out parties with zero mentions
    return partyMentionsDf


def CountPartyDebates(debatesDf, searchTerm):
    """
    Count the number of debates in which each party is mentioned for a specific search term.

    Args:
        debatesDf: DataFrame containing the scraped debates data.
        searchTerm: The search term to filter debates.

    Returns:
        A pandas DataFrame with parties and their debate counts.
    """
    parliamentarians = ScrapeParliamentarianNames()
    partyCounts = {}

    # Initialise party counts
    for group in ["TDs", "Senators"]:
        for member in parliamentarians[group]:
            party = member["party"]
            if party not in partyCounts:
                partyCounts[party] = 0

    # Count debates
    for _, row in debatesDf.iterrows():
        debateMentions = set()
        for i in range(1, 50):  # Iterate through potential debate pages
            htmlText = row.get(f"text_{i}")
            if pd.isna(htmlText):  # Skip if the text is missing
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(htmlText, "html.parser")
            body = soup.body.get_text() if soup.body else ""

            # Check if the search term is in the text
            if searchTerm.lower() in body.lower():
                # Track parties mentioned in this debate
                for group in ["TDs", "Senators"]:
                    for member in parliamentarians[group]:
                        if member["name"].lower() in body.lower():
                            debateMentions.add(member["party"])

        # Update debate counts
        for party in debateMentions:
            partyCounts[party] += 1

    # Convert counts to a DataFrame
    partyDebatesDf = pd.DataFrame(list(partyCounts.items()), columns=["Party", "Debates Mentioned"])
    partyDebatesDf = partyDebatesDf[partyDebatesDf["Debates Mentioned"] > 0]  # Filter out parties with zero debates
    return partyDebatesDf


def CountConstituencyMentions(debatesDf, searchTerm):
    """
    Count the total number of mentions of each constituency for a specific search term.

    Args:
        debatesDf: DataFrame containing the scraped debates data.
        searchTerm: The search term to filter debates.

    Returns:
        A pandas DataFrame with constituencies and their total mention counts.
    """
    parliamentarians = ScrapeParliamentarianNames()
    constituencyCounts = {}

    # Initialise constituency counts
    for group in ["TDs", "Senators"]:
        for member in parliamentarians[group]:
            constituency = member["constituency"]
            if constituency not in constituencyCounts:
                constituencyCounts[constituency] = 0

    # Count mentions
    for _, row in debatesDf.iterrows():
        for i in range(1, 50):  # Iterate through potential debate pages
            htmlText = row.get(f"text_{i}")
            if pd.isna(htmlText):  # Skip if the text is missing
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(htmlText, "html.parser")
            body = soup.body.get_text() if soup.body else ""

            # Check if the search term is in the text
            if searchTerm.lower() in body.lower():
                # Count mentions for each constituency
                for group in ["TDs", "Senators"]:
                    for member in parliamentarians[group]:
                        if member["name"].lower() in body.lower():
                            constituencyCounts[member["constituency"]] += body.lower().count(member["name"].lower())

    # Convert counts to a DataFrame
    constituencyMentionsDf = pd.DataFrame(list(constituencyCounts.items()), columns=["Constituency", "Total Mentions"])
    constituencyMentionsDf = constituencyMentionsDf[constituencyMentionsDf["Total Mentions"] > 0]  # Filter out constituencies with zero mentions
    return constituencyMentionsDf


def CountConstituencyDebates(debatesDf, searchTerm):
    """
    Count the number of debates in which each constituency is mentioned for a specific search term.

    Args:
        debatesDf: DataFrame containing the scraped debates data.
        searchTerm: The search term to filter debates.

    Returns:
        A pandas DataFrame with constituencies and their debate counts.
    """
    parliamentarians = ScrapeParliamentarianNames()
    constituencyCounts = {}

    # Initialise constituency counts
    for group in ["TDs", "Senators"]:
        for member in parliamentarians[group]:
            constituency = member["constituency"]
            if constituency not in constituencyCounts:
                constituencyCounts[constituency] = 0

    # Count debates
    for _, row in debatesDf.iterrows():
        debateMentions = set()
        for i in range(1, 50):  # Iterate through potential debate pages
            htmlText = row.get(f"text_{i}")
            if pd.isna(htmlText):  # Skip if the text is missing
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(htmlText, "html.parser")
            body = soup.body.get_text() if soup.body else ""

            # Check if the search term is in the text
            if searchTerm.lower() in body.lower():
                # Track constituencies mentioned in this debate
                for group in ["TDs", "Senators"]:
                    for member in parliamentarians[group]:
                        if member["name"].lower() in body.lower():
                            debateMentions.add(member["constituency"])

        # Update debate counts
        for constituency in debateMentions:
            constituencyCounts[constituency] += 1

    # Convert counts to a DataFrame
    constituencyDebatesDf = pd.DataFrame(list(constituencyCounts.items()), columns=["Constituency", "Debates Mentioned"])
    constituencyDebatesDf = constituencyDebatesDf[constituencyDebatesDf["Debates Mentioned"] > 0]  # Filter out constituencies with zero debates
    return constituencyDebatesDf


def CreateCaseLevelData(debatesDf, searchTerms):
    """
    Create case-level data with a row for each debate, including counts of mentions of each term, parliamentarian, party, and constituency.

    Args:
        debatesDf: DataFrame containing the scraped debates data.
        searchTerms: A list of search terms to analyse.

    Returns:
        A pandas DataFrame with case-level data.
    """
    parliamentarians = ScrapeParliamentarianNames()
    caseData = []

    for _, row in debatesDf.iterrows():
        debateData = {
            "Date": row["date"].strftime("%Y-%m-%d"),
            "Debate_ID": row.name,  # Use the row index as a unique debate ID
        }

        # Initialise counts
        termCounts = {term: 0 for term in searchTerms}
        parliamentarianCounts = {member["name"]: 0 for group in ["TDs", "Senators"] for member in parliamentarians[group]}
        partyCounts = {member["party"]: 0 for group in ["TDs", "Senators"] for member in parliamentarians[group]}
        constituencyCounts = {member["constituency"]: 0 for group in ["TDs", "Senators"] for member in parliamentarians[group]}

        for i in range(1, 50):  # Iterate through potential debate pages
            htmlText = row.get(f"text_{i}")
            if pd.isna(htmlText):  # Skip if the text is missing
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(htmlText, "html.parser")
            body = soup.body.get_text() if soup.body else ""

            # Count mentions of each search term
            for term in searchTerms:
                termCounts[term] += body.lower().count(term.lower())

            # Count mentions of each parliamentarian, party, and constituency
            for group in ["TDs", "Senators"]:
                for member in parliamentarians[group]:
                    if member["name"].lower() in body.lower():
                        parliamentarianCounts[member["name"]] += body.lower().count(member["name"].lower())
                        partyCounts[member["party"]] += body.lower().count(member["name"].lower())
                        constituencyCounts[member["constituency"]] += body.lower().count(member["name"].lower())

        # Add counts to the debate data
        debateData.update(termCounts)
        debateData.update({f"Parliamentarian_{name}": count for name, count in parliamentarianCounts.items()})
        debateData.update({f"Party_{party}": count for party, count in partyCounts.items()})
        debateData.update({f"Constituency_{constituency}": count for constituency, count in constituencyCounts.items()})

        caseData.append(debateData)

    # Convert case data to a DataFrame
    caseLevelDf = pd.DataFrame(caseData)
    return caseLevelDf


def DebScraper(startDate=datetime(2016, 7, 1), endDate=datetime(2022, 2, 28), searchTerms="CervicalCheck", regexPattern=None, delay=None):
    """
    Main function to execute the script.

    Args:
        startDate: Start date for scraping debates.
        endDate: End date for scraping debates.
        searchTerms: A single search term (string) or a Boolean expression (e.g., "health AND medical OR wellness").
        regexPattern: Regex pattern to match (default: None).
        delay: Delay between page loads (None for no delay, True for default delay of 2 seconds, or an integer for custom delay).
    """
    # Calculate the total number of pages to be accessed
    totalDays = (endDate - startDate).days + 1

    # Warn the user if the search involves a large number of pages
    if totalDays * 50 > 1000:  # Assuming up to 50 pages per day
        print(
            f"WARNING: Your search involves accessing approximately {totalDays * 50} pages. "
            "Consider adding a delay between requests (e.g., 2–5 seconds) or breaking your search "
            "into smaller date ranges and running them over separate VPNs to avoid anti-bot measures."
        )

    # Inform the user about how to abort the search
    print("\nNOTE: You can abort the search at any time by pressing Ctrl+C.")
    print("If you abort, the progress will be saved, including scraped debates and processed text files.\n")

    # Validate the dates
    ValidateDates(startDate, endDate)

    # Set up the WebDriver
    driver = SetupWebDriver()

    try:
        # Scrape debates for the specified date range
        print(f"Scraping debates from {startDate} to {endDate}...")
        debatesDf = ScrapeDebates(driver, startDate, endDate, delay, searchTerms, regexPattern)

        # Save raw data for debugging or future use
        debatesDf.to_pickle('debates_data.pkl')

        # Process and save debates mentioning the search terms
        print(f"Processing and saving debates mentioning the specified terms...")
        ProcessAndSaveTexts(debatesDf, outputDir="ProcessedDebates", searchTerms=searchTerms, regexPattern=regexPattern)
        print("\nProcessing complete. Files saved in 'ProcessedDebates' directory.")

    except KeyboardInterrupt:
        # Handle manual abort (Ctrl+C) gracefully
        print("\nSearch aborted by user. Exiting gracefully...")
    finally:
        # Quit the WebDriver to free resources with extra error handling
        if 'driver' in locals():
            try:
                driver.quit()
            except Exception:
                # Silently ignore any exceptions during driver cleanup
                pass


def ScheduleScraping(interval, startDate, endDate, searchTerms):
    """
    Schedule the scraping process to run at regular intervals.

    Args:
        interval: Interval in minutes to run the scraping process.
        startDate: Start date for scraping.
        endDate: End date for scraping.
        searchTerms: Search terms to look for.
    """
    def job():
        DebScraper(startDate=startDate, endDate=endDate, searchTerms=searchTerms)
        logging.info("Scheduled scraping job executed.")

    schedule.every(interval).minutes.do(job)
    print(f"Scraping scheduled every {interval} minutes.")
    while True:
        schedule.run_pending()
        time.sleep(1)


def main():
    """
    Main function to parse command-line arguments and execute the requested functionality.

    Usage (command line):
        python OireachtasDebatesScraper.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --search-terms "term1 AND term2" --delay 2

    For more details, see the README or the JOSS paper.
    """
    import sys

    try:
        # Use argparse to parse command-line arguments
        parser = argparse.ArgumentParser(
            description="Scrape Dáil debates from the Oireachtas website, process the text for mentions of specified search terms, and save the results."
        )

        # Add arguments for scraping debates
        parser.add_argument(
            "--start-date",
            type=str,
            default="2016-07-01",  # Default start date
            help="Start date for scraping debates (format: YYYY-MM-DD). Default is 2016-07-01.",
        )
        parser.add_argument(
            "--end-date",
            type=str,
            default="2022-02-28",  # Default end date
            help="End date for scraping debates (format: YYYY-MM-DD). Default is 2022-02-28.",
        )
        parser.add_argument(
            "--search-terms",
            type=str,
            default="CervicalCheck",
            help="Search terms to look for in debates (default: 'CervicalCheck').",
        )
        parser.add_argument(
            "--regex-pattern",
            type=str,
            default=None,
            help="Regex pattern to match in debates (default: None).",
        )
        parser.add_argument(
            "--delay",
            type=int,
            default=None,
            help="Delay between page loads in seconds (default: None).",
        )

        # Parse the arguments
        args = parser.parse_args()

        # Convert date strings to datetime objects
        startDate = datetime.strptime(args.start_date, "%Y-%m-%d")
        endDate = datetime.strptime(args.end_date, "%Y-%m-%d")

        # Calculate the total number of pages to be accessed
        totalDays = (endDate - startDate).days + 1
        estimatedPages = totalDays * 50  # Assuming up to 50 pages per day

        # Display warning if the search involves a large number of pages
        if estimatedPages > 1000:
            print(
                f"WARNING: Your search involves accessing approximately {estimatedPages} pages. "
                "Consider adding a delay between requests (e.g., 2–5 seconds) or breaking your search "
                "into smaller date ranges and running them over separate VPNs to avoid anti-bot measures."
            )

        # Inform the user about how to abort the search
        print("\nNOTE: You can abort the search at any time by pressing Ctrl+C.")
        print("If you abort, the progress will be saved, including scraped debates and processed text files.\n")

        # Scrape debates for the specified date range
        driver = SetupWebDriver()
        ValidateDates(startDate, endDate)
        debatesDf = ScrapeDebates(
            driver=driver,
            startDate=startDate,
            endDate=endDate,
            delay=args.delay,
            searchTerms=args.search_terms,
            regexPattern=args.regex_pattern,
        )

        # Save raw data for debugging or future use
        debatesDf.to_pickle('debates_data.pkl')

        # Process and save debates mentioning the search terms
        ProcessAndSaveTexts(debatesDf, outputDir="ProcessedDebates", searchTerms=args.search_terms, regexPattern=args.regex_pattern)
        print("Processing complete. Files saved in 'ProcessedDebates' directory.")

    except KeyboardInterrupt:
        # Handle manual abort (Ctrl+C) gracefully
        print("\nSearch aborted by user. Exiting gracefully...")
    except Exception as e:
        # Handle unexpected errors
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Simplified driver cleanup
        if 'driver' in locals():
            try:
                driver.quit()
            except:
                pass


if __name__ == "__main__":
    # Only run main if executed as a script, not on import
    main()