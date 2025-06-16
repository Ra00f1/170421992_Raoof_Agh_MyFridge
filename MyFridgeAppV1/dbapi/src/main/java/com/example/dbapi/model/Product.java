package com.example.dbapi.model;

import jakarta.persistence.*;

@Entity
public class Product 
{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;
    private String category;
    private double price;

    @Column(name = "count_type")
    private String countType; // kg, lt, adet gibi

    // Constructor
    public Product() {}

    public Product(String name, String category, double price, String countType) 
    {
        this.name = name;
        this.category = category;
        this.price = price;
        this.countType = countType;
    }

    // Getter & Setter
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }

    public double getPrice() { return price; }
    public void setPrice(double price) { this.price = price; }

    public String getCountType() { return countType; }
    public void setCountType(String countType) { this.countType = countType; }
}
