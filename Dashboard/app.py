from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE=Path(__file__).resolve().parent.parent; TARGET="daily_rate"
st.set_page_config("Rental Intelligence | Vehicle Pricing","🚗",layout="wide")
@st.cache_resource
def model(): return joblib.load(BASE/"Models"/"vehicle_rental_price_pipeline.joblib")
@st.cache_data
def js(name):
 with (BASE/"Models"/name).open() as f:return json.load(f)
@st.cache_data
def data():
 f=next((BASE/"DataSet").glob("*.csv")); raw=pd.read_csv(f)
 d=raw.rename(columns={"fuelType":"fuel_type","renterTripsTaken":"renter_trips_taken","reviewCount":"review_count","location.city":"location_city","location.country":"location_country","location.latitude":"location_latitude","location.longitude":"location_longitude","location.state":"location_state","rate.daily":TARGET,"vehicle.make":"vehicle_make","vehicle.model":"vehicle_model","vehicle.type":"vehicle_type","vehicle.year":"vehicle_year"}).copy()
 d["vehicle_age"]=(2020-d.vehicle_year).clip(lower=0);d["has_rating"]=d.rating.notna().astype(int);d.fuel_type=d.fuel_type.fillna("UNKNOWN")
 return raw,d
def title(a,b):st.subheader(a);st.caption(b)
def usd(x):return f"${x:,.0f}"
def corr_color(v):
 if pd.isna(v):return ""
 neg,mid,pos=(217,83,79),(255,255,255),(31,119,180)
 f=1+v if v<=0 else v;end=mid if v<=0 else pos;start=neg if v<=0 else mid
 r,g,b=(int(start[i]+(end[i]-start[i])*f) for i in range(3))
 return f"background-color: rgb({r},{g},{b})"
def filtered(d):
 st.sidebar.markdown("### Analysis filters");st.sidebar.caption("Use the price range to focus the analytical views.")
 r=st.sidebar.slider("Daily rate range (USD)",int(d[TARGET].min()),int(d[TARGET].max()),(int(d[TARGET].min()),int(d[TARGET].max())))
 return d[d[TARGET].between(*r)]
def empty(d):
 if d.empty:st.warning("No listings match the current filters.");return True
 return False
def overview(d,all):
 st.markdown("## 📊 Dataset Overview");st.caption("Explore historical rental listings, understand pricing patterns, and estimate a daily rate.")
 if empty(d):return
 c=st.columns(4);avg=d[TARGET].mean();c[0].metric("Listings in view",f"{len(d):,}",f"of {len(all):,} total");c[1].metric("Average daily rate",usd(avg),f"{avg-all[TARGET].mean():+.0f} vs. all");c[2].metric("Median daily rate",usd(d[TARGET].median()));c[3].metric("Rated listings",f"{d.has_rating.mean():.0%}")
 a,b=st.columns(2)
 with a:
  title("Daily-rate distribution","Historical price frequency across the selected rate range.");hist=pd.cut(d[TARGET],bins=20).value_counts().sort_index();hist.index=hist.index.astype(str);st.bar_chart(hist,color="#1f77b4")
 with b:
  title("Listing activity","Average daily rate grouped by renter trip history.");activity=d.assign(trip_band=pd.cut(d.renter_trips_taken,[-1,0,5,20,50,100,float("inf")],labels=["0","1–5","6–20","21–50","51–100","100+"])).groupby("trip_band",observed=False)[TARGET].mean();st.bar_chart(activity,color="#1f77b4")
 title("Reading the data","The average shows the overall level, while the median is less affected by unusually high-priced listings. Use the price-range control to explore the distribution.")
 st.info("The dashboard uses historical listing data. Estimates and patterns should be checked against current market conditions before setting a public rental rate.")
def data_page(raw,d):
 st.title("Data overview");st.write("Dataset scope, quality, and the records in your current selection.")
 c=st.columns(4);c[0].metric("Rows in view",f"{len(d):,}");c[1].metric("Source features",raw.shape[1]);c[2].metric("Missing cells (source)",f"{raw.isna().sum().sum():,}");c[3].metric("States represented",d.location_state.nunique())
 title("Data quality","Missing values are calculated from the source dataset; the model pipeline preprocesses its inputs.")
 m=raw.isna().sum().rename("missing_values").reset_index().rename(columns={"index":"feature"});m["missing_rate"]=m.missing_values/len(raw);st.dataframe(m.sort_values("missing_values",ascending=False),hide_index=True,width="stretch",column_config={"missing_rate":st.column_config.NumberColumn("Missing rate",format="%.1f%%")})
 with st.expander("Feature dictionary"):st.dataframe(pd.DataFrame([("daily_rate","Target","Daily listing price in USD"),("vehicle_*","Vehicle","Make, model, type and year"),("rating, review_count, renter_trips_taken","Activity","Rating and historical activity"),("location_*","Location","City, state and coordinates"),("fuel_type","Vehicle","Reported fuel category")],columns=["Feature(s)","Category","Description"]),hide_index=True,width="stretch")
 cols=["vehicle_make","vehicle_model","vehicle_type","fuel_type","vehicle_year","location_state","rating","renter_trips_taken",TARGET];title("Filtered listing data","Download the current selection for further review.");st.download_button("Download filtered data (CSV)",d[cols].to_csv(index=False).encode(),"vehicle_rental_filtered_data.csv","text/csv");st.dataframe(d[cols],hide_index=True,width="stretch",height=340)
def eda(d):
 st.title("Exploratory analysis");st.write("Interactive views of the historical patterns behind rental pricing.")
 if empty(d):return
 a,b=st.columns(2)
 with a:title("Daily-rate distribution","Most listings sit below a smaller high-price tail.");hist=pd.cut(d[TARGET],bins=30).value_counts().sort_index();hist.index=hist.index.astype(str);st.bar_chart(hist,color="#1f77b4")
 with b:title("Rate by vehicle type","Average rate, by vehicle type.");st.bar_chart(d.groupby("vehicle_type")[TARGET].mean().sort_values(ascending=False),color="#1f77b4")
 hue=st.selectbox("Segment scatter points by",["vehicle_type","fuel_type","location_state"]);title("Trips and daily rate","Each point is a listing; colour shows the selected segment.")
 sample=d.sample(min(len(d),2000),random_state=42);st.scatter_chart(sample,x="renter_trips_taken",y=TARGET,color=hue,size=20)
 title("Numeric correlations","Pearson correlations show linear association only, not causation.");n=[TARGET,"rating","renter_trips_taken","review_count","vehicle_year","vehicle_age","location_latitude","location_longitude"];st.dataframe(d[n].corr().style.format("{:.2f}").map(corr_color),width="stretch")
def performance(m,meta):
 st.title("Model performance");st.write("Evaluation details available with the saved regression model. Classification metrics do not apply.")
 p=meta["performance"];c=st.columns(3);c[0].metric("Mean absolute error",f"${p['mae']:,.2f}");c[1].metric("Root mean squared error",f"${p['rmse']:,.2f}");c[2].metric("R² score",f"{p['r2']:.4f}");st.success(f"Selected model: **{meta['model_name']}**");st.caption("MAE is typical absolute error in dollars; RMSE weights larger errors more heavily; R² is captured variation in the recorded evaluation.")
 features=["rating","renter_trips_taken","location_latitude","location_longitude","has_rating","vehicle_age","fuel_type","location_state","vehicle_make","vehicle_type"];names=m.named_steps["preprocessor"].get_feature_names_out();imp=pd.DataFrame({"encoded":names,"importance":m.named_steps["model"].feature_importances_});imp["feature"]=imp.encoded.apply(lambda x:next((f for f in features if x.endswith("__"+f)or x.startswith("categorical__"+f+"_")),"Other"));title("Feature importance","Derived from the fitted estimator; importance is model reliance, not causation.");st.bar_chart(imp.groupby("feature").importance.sum().sort_values(ascending=False),color="#1f77b4")
 with st.expander("Model configuration and evaluation coverage"):st.json({"input_features":meta["input_features"],"best_hyperparameters":meta["best_hyperparameters"]});st.info("Only one saved model and aggregate regression metrics are available. Held-out actual-versus-predicted records and alternative model results were not supplied, so comparison and residual charts are intentionally not shown.")
def predict(m,o):
 st.title("Price estimator");st.write("Enter listing details to estimate a daily rental price.")
 with st.form("prediction"):
  a,b=st.columns(2)
  with a:st.markdown("#### Vehicle");make=st.selectbox("Make",o["vehicle_make"]);typ=st.selectbox("Type",o["vehicle_type"]);fuel=st.selectbox("Fuel type",o["fuel_type"]);age=st.number_input("Vehicle age (years)",o["vehicle_age"]["minimum"],o["vehicle_age"]["maximum"],o["vehicle_age"]["default"],step=1)
  with b:
   st.markdown("#### Listing and location");status=st.radio("Rating status",["Rated","Unrated"],horizontal=True);has=int(status=="Rated");rating=st.slider("Vehicle rating",1.,5.,4.98,.01,disabled=not has)if has else float("nan");trips=st.number_input("Renter trips taken",o["renter_trips_taken"]["minimum"],o["renter_trips_taken"]["maximum"],o["renter_trips_taken"]["default"],step=1);state=st.selectbox("State",o["location_state"]);x,y=st.columns(2);lat=x.number_input("Latitude",o["location_latitude"]["minimum"],o["location_latitude"]["maximum"],o["location_latitude"]["default"],format="%.4f");lon=y.number_input("Longitude",o["location_longitude"]["minimum"],o["location_longitude"]["maximum"],o["location_longitude"]["default"],format="%.4f")
  go=st.form_submit_button("Estimate daily rental price",type="primary",width="stretch")
 if go:
  row=pd.DataFrame({"fuel_type":[fuel],"rating":[rating],"renter_trips_taken":[trips],"location_latitude":[lat],"location_longitude":[lon],"location_state":[state],"vehicle_make":[make],"vehicle_type":[typ],"has_rating":[has],"vehicle_age":[age]}).reindex(columns=m.feature_names_in_)
  try:
   v=float(m.predict(row)[0])
   st.markdown(f"<div class='price-result'><div class='price-result-label'>Estimated daily rental price</div><div class='price-result-value'>${v:,.2f}</div><div class='price-result-caption'>Historical-data model estimate — not a guaranteed market price.</div></div>",unsafe_allow_html=True)
   st.dataframe(row,hide_index=True,width="stretch")
  except Exception as e:st.error(f"Prediction failed: {e}")
def insights(d):
 st.title("Insights & recommendations");st.write("Historical observations and operational guidance, clearly separated.")
 if empty(d):return
 g=d.groupby("vehicle_type")[TARGET].mean().sort_values(ascending=False);state=d.groupby("location_state")[TARGET].mean().idxmax();a,b=st.columns(2)
 with a:st.markdown("#### Observations");st.write(f"- Median daily rate: **{usd(d[TARGET].median())}** across **{len(d):,}** listings.");st.write(f"- **{g.index[0].title()}** has the highest selected average ({usd(g.iloc[0])}).");st.write(f"- **{state}** has the highest selected-state average rate.")
 with b:st.markdown("#### Recommendations");st.write("- Use the estimator as a starting point, then validate against current local supply and demand.");st.write("- Compare medians and distributions by vehicle type instead of pricing from one outlier.");st.write("- Refresh analysis with newer booking and competitor data; this historical dataset can become stale.")
def main():
 st.markdown("""<style>
:root{--accent:#1f77b4;--accent-dark:#155a8a}.stApp{background:#0E1117;color:#E5E7EB}.block-container{max-width:1120px;padding:3.25rem 3.5rem 3rem}h1,h2,h3{color:#F8FAFC!important;letter-spacing:-.02em}.dashboard-hero{padding:1.1rem 0 2.4rem;margin:0;text-align:center;background:none;color:#94A3B8;box-shadow:none}.dashboard-hero h1{font-size:2rem!important;font-weight:750;color:var(--accent)!important;margin:0 0 .65rem}.dashboard-hero p{max-width:720px;margin:auto;font-size:.92rem;line-height:1.65;color:#94A3B8}.dashboard-hero .eyebrow{display:none}[data-testid='stMetric']{background:#F8FAFC;border:1px solid #D8DEE8;border-top:3px solid var(--accent);border-radius:12px;padding:1.2rem 1rem;box-shadow:0 4px 14px rgba(0,0,0,.22)}[data-testid='stMetricLabel']{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:#64748B!important}[data-testid='stMetricValue']{font-size:1.45rem;color:#1E293B!important}section[data-testid='stSidebar']{background:#262730;border-right:1px solid #30323D}section[data-testid='stSidebar'] *{color:#F8FAFC!important}section[data-testid='stSidebar'] h1{font-size:1.15rem!important;margin-top:1.3rem}section[data-testid='stSidebar'] label{font-size:.9rem!important;padding:.28rem 0}section[data-testid='stSidebar'] [data-testid='stWidgetLabel']{color:#B8C1CF!important}.stButton>button,.stFormSubmitButton>button{background:var(--accent)!important;border:0!important;border-radius:8px!important;color:#fff!important;font-weight:650!important}.stButton>button:hover,.stFormSubmitButton>button:hover{background:var(--accent-dark)!important}.stCaption,p,li{color:#AAB4C3!important}.stAlert{border-radius:8px}[data-testid='stDataFrame']{border:1px solid #303744;border-radius:8px;overflow:hidden}.dashboard-footer{margin-top:2.5rem;padding:1.25rem;text-align:center;background:#171B24;border-top:2px solid var(--accent);border-radius:8px;color:#9AA5B5;font-size:.85rem}.dashboard-footer strong{color:#F8FAFC}.price-result{background:var(--accent);border-radius:14px;padding:1.9rem 1.5rem;text-align:center;box-shadow:0 10px 28px rgba(31,119,180,.35);margin:.75rem 0 1.4rem}.price-result-label{color:#EAF6FF!important;font-size:.8rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.5rem}.price-result-value{color:#fff!important;font-size:3rem;font-weight:800;line-height:1.1;letter-spacing:-.02em}.price-result-caption{color:#EAF6FF!important;font-size:.82rem;margin-top:.6rem;opacity:.95}@media(max-width:800px){.block-container{padding:1.4rem}.dashboard-hero h1{font-size:1.6rem!important}}</style>""",unsafe_allow_html=True)
 try:m=model();meta=js("model_metadata.json");o=js("dashboard_input_options.json");raw,all=data()
 except Exception as e:st.error(f"Application startup failed: {e}");st.stop()
 st.sidebar.title("Rental Intelligence");st.sidebar.caption("Vehicle rental analytics & pricing");page=st.sidebar.radio("Navigate",["Overview","Data Overview","Exploratory Analysis","Model Performance","Price Estimator","Insights & Recommendations"]);d=filtered(all);st.sidebar.divider();st.sidebar.caption("Source: CarRentalData • Historical US listings");st.markdown(f"<div class='dashboard-hero'><div class='eyebrow'>Vehicle rental intelligence</div><h1>{page}</h1><p>Historical listing analytics and model-assisted daily-rate estimates.</p></div>",unsafe_allow_html=True)
 {"Overview":lambda:overview(d,all),"Data Overview":lambda:data_page(raw,d),"Exploratory Analysis":lambda:eda(d),"Model Performance":lambda:performance(m,meta),"Price Estimator":lambda:predict(m,o),"Insights & Recommendations":lambda:insights(d)}[page]();st.markdown("<div class='dashboard-footer'><strong>Rental Intelligence</strong> &nbsp;•&nbsp; Historical vehicle rental analytics &nbsp;•&nbsp; Model-assisted estimates</div>",unsafe_allow_html=True)
if __name__=="__main__":main()







