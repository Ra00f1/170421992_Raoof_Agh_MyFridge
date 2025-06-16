package com.example.llmapi.model;

public class Recipe 
{
    private String adi;
    private String icerigi;
    private String yapilisi;

    // Constructor
    public Recipe(String adi, String icerigi, String yapilisi) 
    {
        this.adi = adi;
        this.icerigi = icerigi;
        this.yapilisi = yapilisi;
    }
    // Default constructor
    public Recipe() 
    {
        this.adi = "";
        this.icerigi = "";
        this.yapilisi = "";
    }

    // Getter - Setter
    public String getAdi() 
    { 
        return adi; 
    }
    public void setAdi(String adi) 
    { 
        this.adi = adi; 
    }

    public String getIcerigi() 
    { 
        return icerigi; 
    }
    public void setIcerigi(String icerigi) 
    { 
        this.icerigi = icerigi; 
    }

    public String getYapilisi() 
    { 
        return yapilisi; 
    }
    public void setYapilisi(String yapilisi) 
    { 
        this.yapilisi = yapilisi; 
    }

    @Override
    public String toString() 
    {
        return "Recipe{" +
                "adi='" + adi + '\'' +
                ", icerigi='" + icerigi + '\'' +
                ", yapilisi='" + yapilisi + '\'' +
                '}';
    }
}

