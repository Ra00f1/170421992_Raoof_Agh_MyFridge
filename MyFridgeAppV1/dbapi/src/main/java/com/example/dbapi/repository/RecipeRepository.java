package com.example.dbapi.repository;

import com.example.dbapi.model.Recipe;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface RecipeRepository extends JpaRepository<Recipe, Long> 
{
    Optional<Recipe> findByYemekAdi(String yemekAdi);
}
