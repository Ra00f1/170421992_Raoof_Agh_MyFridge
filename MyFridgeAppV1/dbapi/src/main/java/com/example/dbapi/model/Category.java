package com.example.dbapi.model;

import jakarta.persistence.*;

@Entity
@Table(name = "category")
public class Category 
{

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "name")
    private String name;

    public Category(String name) 
    {
    this.name = name;
}
    public Category() {
        // JPA için varsayılan yapıcı
    }
    // Getter ve Setter’ları unutma

    public String getName() 
    {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    // id için getter-setter zaten vardır
}
