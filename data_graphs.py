# -*- coding: utf-8 -*-


import seaborn as sb
import matplotlib.pyplot as plt


from clean_data import get_clean_df

print('Loading data and showing graphs ...','\n\n')

#df = get_clean_df()[1]

#print(df.columns)


def print_graphs(df):
           
    fig, axs = plt.subplots(2,1, figsize=(16,8))    
        
    sb.countplot(x='Ethnicity',data=df,hue='less than 50K',palette='Paired',ax=axs[0])
    sb.countplot(x='Gender',data=df,hue='less than 50K',palette='rocket',ax=axs[1])
    
    new_labels = ['more than 50K', 'less than 50K']
    
    axs[0].legend(labels=new_labels)
    axs[1].legend(labels=new_labels)
    
    bachelors = sb.factorplot('Bachelors',data=df,kind="count",hue='less than 50K',aspect=2.5,legend=True,legend_out=True)
    bachelors.set_xticklabels(rotation=30)    
    for t, l in zip(bachelors._legend.texts, new_labels): t.set_text(l)    
    
    fam_rel = sb.factorplot('Relationship',data=df,kind="count",hue='less than 50K',aspect=2.5,palette='Set2')
    fam_rel.set_xticklabels(rotation=30)
    for t, l in zip(fam_rel._legend.texts, new_labels): t.set_text(l)
    
    fig.tight_layout()

    fig.show()
    
#======================================
    
  
 
