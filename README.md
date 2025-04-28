# OireachtasDebatesScraper

OireachtasDebatesScraper is a Python tool for scraping and analyzing Dáil debates from the Oireachtas website. It supports Boolean logic, regex searches, and text processing for advanced analysis.

## Features
- Scrapes debates for a specified date range.
- Identifies and processes debates mentioning specified search terms.
- Supports Boolean logic (AND/OR/NOT) and regex searches.
- Removes duplicates and cleans text.
- Saves processed text to individual `.txt` files.
- Includes frequency, co-occurrence, and metadata analysis.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/weinerrb/OireachtasDebatesScraper.git
   cd OireachtasDebatesScraper


2. Install the required Python libraries:
pip install -r requirements.txt

3. Ensure chromedriver is installed and its path is correctly configured.

## Usage
Run the scraper using the following command:
python OireachtasDebatesScraper.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --search-terms "term1 AND term2" --delay 2

## Example
To scrape debates from January 1, 2020, to December 31, 2020, for mentions of "health AND wellness" with a 2-second delay:
python OireachtasDebatesScraper.py --start-date 2020-01-01 --end-date 2020-12-31 --search-terms "health AND wellness" --delay 2

Here’s a rewritten version of your README.md as a single, cohesive file:

```markdown
# OireachtasDebatesScraper

OireachtasDebatesScraper is a Python tool designed to scrape and analyze Dáil debates from the Oireachtas website. It supports advanced text processing features, including Boolean logic, regex searches, and metadata analysis, making it a powerful tool for researchers and analysts.

## Features
- Scrape debates for a specified date range.
- Identify and process debates mentioning specific search terms.
- Support for Boolean logic (AND/OR/NOT) and regex-based searches.
- Remove duplicate entries and clean text for analysis.
- Save processed text to individual `.txt` files.
- Perform frequency, co-occurrence, and metadata analysis.

## Installation
### Prerequisites
- Python 3.8 or higher
- Google Chrome and [ChromeDriver](https://chromedriver.chromium.org/) (ensure the version matches your Chrome browser)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/weinerrb/OireachtasDebatesScraper.git
   cd OireachtasDebatesScraper
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `chromedriver` is installed and its path is correctly configured.

## Usage
Run the scraper using the following command:
```bash
python OireachtasDebatesScraper.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --search-terms "term1 AND term2" --delay 2
```

### Example
To scrape debates from January 1, 2020, to December 31, 2020, for mentions of "health AND wellness" with a 2-second delay:
```bash
python OireachtasDebatesScraper.py --start-date 2020-01-01 --end-date 2020-12-31 --search-terms "health AND wellness" --delay 2
```

### Command-Line Arguments
- `--start-date`: Start date for scraping debates (format: `YYYY-MM-DD`). Default: `2016-07-01`.
- `--end-date`: End date for scraping debates (format: `YYYY-MM-DD`). Default: `2022-02-28`.
- `--search-terms`: Search terms to look for in debates (supports Boolean logic). Default: `"CervicalCheck"`.
- `--regex-pattern`: Regex pattern to match in debates. Default: `None`.
- `--delay`: Delay between page loads in seconds. Default: `None`.

## Output
- **Raw Data**: Saved as `debates_data.pkl` for debugging or future use.
- **Processed Texts**: Saved in the `ProcessedDebates/` directory as `.txt` files.
- **Metadata**: Optionally exported in CSV or JSON format.

## Dependencies
The following Python libraries are required:
- `selenium`
- `pandas`
- `beautifulsoup4`
- `matplotlib`
- `seaborn`
- `wordcloud`
- `textblob`
- `requests`
- `schedule`

Install them using:
```bash
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
If you use this software in your research, please cite it as:
```
Ruairí Weiner. (2025). OireachtasDebatesScraper: A Python tool for scraping and analyzing Irish parliamentary debates.
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## Contact
For questions or support, please contact Ruairí Weiner at weinerr@tcd,ie
```
