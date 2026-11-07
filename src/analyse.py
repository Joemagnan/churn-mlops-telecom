"""Analyses complémentaires pour le rapport (requirements-dev.txt) : courbes, seuil, variables.

Le seuil est étudié sur le jeu de VALIDATION. Le jeu test ne sert qu'à tracer les
courbes du modèle final, déjà évalué une seule fois par src.train.
"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (RocCurveDisplay, PrecisionRecallDisplay, precision_score,
                             recall_score, f1_score)
from src.config import DATA_PATH, MODEL_PATH, ROOT
from src.train import load_data, split_data

REPORTS=ROOT/"reports"

def threshold_table(model, X, y):
    score=model.predict_proba(X)[:,1]
    rows=[]
    for t in np.round(np.arange(.30,.76,.05),2):
        pred=(score>=t).astype(int)
        rows.append({"seuil":float(t),"precision":round(float(precision_score(y,pred,zero_division=0)),4),
                     "recall":round(float(recall_score(y,pred)),4),"f1":round(float(f1_score(y,pred)),4),
                     "clients_alertes":int(pred.sum())})
    return rows

def top_variables(model, n=8):
    names=model[:-1].get_feature_names_out()
    est=model[-1]
    weights=est.coef_[0] if hasattr(est,"coef_") else est.feature_importances_
    order=np.argsort(weights)
    def pick(indices):
        return [{"variable":str(names[i]).split("__",1)[1],"poids":round(float(weights[i]),3)} for i in indices]
    return {"hausse_du_risque":pick(order[::-1][:n]),"baisse_du_risque":pick(order[:n])}

def main():
    X,y=load_data(DATA_PATH)
    Xtrain,Xtest,ytrain,ytest,Xa,Xval,ya,yval=split_data(X,y)
    final=joblib.load(MODEL_PATH)
    # Même configuration que le modèle final, réentraînée sur la seule partie apprentissage
    on_train_part=clone(final).fit(Xa,ya)
    result={"seuils_validation":threshold_table(on_train_part,Xval,yval),
            "variables":top_variables(final),
            "taux_churn_test":round(float(ytest.mean()),4)}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS/"analyse.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")

    label={"LogisticRegression":"Régression logistique","RandomForestClassifier":"Forêt aléatoire"}[type(final[-1]).__name__]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,3.8))
    RocCurveDisplay.from_estimator(final,Xtest,ytest,ax=ax1,name=label,curve_kwargs={"color":"#1f4e79"})
    ax1.plot([0,1],[0,1],ls="--",color="grey",lw=1,label="Hasard")
    ax1.set_title("Courbe ROC (jeu test)"); ax1.set_xlabel("Taux de faux positifs"); ax1.set_ylabel("Taux de vrais positifs")
    ax1.legend(loc="lower right",fontsize=8)
    PrecisionRecallDisplay.from_estimator(final,Xtest,ytest,ax=ax2,name=label,curve_kwargs={"color":"#1f4e79"})
    ax2.axhline(ytest.mean(),ls="--",color="grey",lw=1,label=f"Taux de churn ({ytest.mean():.1%})")
    ax2.set_title("Courbe précision-rappel (jeu test)"); ax2.set_xlabel("Rappel"); ax2.set_ylabel("Précision")
    ax2.legend(loc="upper right",fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS/"courbes_roc_pr.png",dpi=160)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
