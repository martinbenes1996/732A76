
# import
import pandas as pd
import unidecode

# cache access
import pkg_resources
CACHE = pkg_resources.resource_filename(__name__, "data/NUTS.csv")
CACHE2 = pkg_resources.resource_filename(__name__, "data/NUTS2.csv")
x = None

def nuts(df):
    """df = pd.DataFrame(data = {'city': ['Brno'], 'country': ['CZ']})"""
    global x
    if x is None:
        x = pd.read_csv(CACHE)
        x2 = pd.read_csv(CACHE2)
        x = x.append(x2)
    def _decode(s):
        try: return unidecode.unidecode(s).lower()
        except: return None
    df = df.apply(_decode)
    print(df)
    x = x[x.isin({"country": df.country.tolist()})]
    df = x["LAU NAME NATIONAL"].apply(_decode).join(df, on = "LAU NAME NATIONAL")
    print(df)
    return
    try:
        nuts3 = line.reset_index(drop = True).at[0, 'NUTS 3 CODE']
        lau = line.reset_index(drop = True).at[0, 'LAU CODE']
        #return nuts3[:-1],nuts3#, lau
        return nuts3,lau
    except:
        return None,None
__all__ = ["nuts"]