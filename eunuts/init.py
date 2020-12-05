
# imports
import pandas as pd
import requests
import tempfile

# cache access
import pkg_resources
CACHE = pkg_resources.resource_filename(__name__, "data/NUTS.csv")
# sources
SOURCES = {
    'current': 'https://ec.europa.eu/eurostat/documents/345175/501971/EU-28-LAU-2019-NUTS-2016.xlsx',
    '2013':    'https://ec.europa.eu/eurostat/documents/345175/501971/EU-28_LAU_2017_NUTS_2013.xlsx'
}
# country tabs
COUNTRIES = [
    "BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR","IT","CY","LV","LT","LU","HU",
    "MT","NL","AT","PL","PT","RO","SI","SK","FI","SE","UK","IS","LI","CH","MK","AL","TR"
]

def init_cache(version = 'current'):
    url = SOURCES[version]
    
    #
    df = None
    with tempfile.TemporaryFile() as fd:
        xls = requests.get(url)
        fd.write(xls.content)
        fd.seek(0)
        
        for country in COUNTRIES:
            print("read", country)
            x = pd.read_excel(url, sheet_name = country)
            fd.seek(0)
        
            x["country"] = country
            if df is None: df = x
            else: df = df.append(x)
        
    df.to_csv(CACHE, index = False)


__all__ = ["init_cache"]