package com.example.dbapi.repository;

import com.example.dbapi.model.UserRecipe;
import com.example.dbapi.model.Recipe;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRecipeRepository extends JpaRepository<UserRecipe, Long> 
{
    Optional<UserRecipe> findByUserIdAndRecipe(Long userId, Recipe recipe);
    //Kullanıcının recipe'yi kullanma sayısını güncellemek için
    Optional<UserRecipe> findByUserIdAndRecipeId(Long userId, Long recipeId);
}