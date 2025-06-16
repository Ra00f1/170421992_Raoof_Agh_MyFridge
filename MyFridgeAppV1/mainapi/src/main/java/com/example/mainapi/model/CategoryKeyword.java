package com.example.mainapi.model;

import jakarta.persistence.*;

@Entity
public class CategoryKeyword 
{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String keyword;        // Örn: "yoğurt", "cola", "piliç" aranacak kısım
    private String categoryName;   // Örn: "süt ürünleri", "içecek", "et ürünleri" kategorizasyonları
    // Bu sınıf, ürünlerin kategorilere göre anahtar kelimelerle eşleştirilmesini sağlar.
    // Örn: "süt ürünleri" kategorisinde "yoğurt" anahtar kelimesi bulunabilir.

    // Constructors
    public CategoryKeyword() 
    {
        // Default constructor
    }

    public CategoryKeyword(String keyword, String categoryName) 
    {
        this.keyword = keyword;
        this.categoryName = categoryName;
    }

    // Getters and Setters
    public Long getId() 
    {
        return id;
    }

    public String getKeyword() 
    {
        return keyword;
    }

    public void setKeyword(String keyword) 
    {
        this.keyword = keyword;
    }

    public String getCategoryName() 
    {
        return categoryName;
    }

    public void setCategoryName(String categoryName) 
    {
        this.categoryName = categoryName;
    }
}

