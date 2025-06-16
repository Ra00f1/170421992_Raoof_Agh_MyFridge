package com.example.dbapi.controller;

import com.example.dbapi.model.Recipe;
import com.example.dbapi.service.RecipeService;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;


@RestController
@RequestMapping("/api/recipe")
public class RecipeController 
{

    private final RecipeService recipeService;

    public RecipeController(RecipeService recipeService) 
    {
        this.recipeService = recipeService;
    }

    @PostMapping("/add/{userId}")
    public ResponseEntity<?> addRecipeToUser(@PathVariable Long userId, @RequestBody Recipe recipe) 
    {
        Recipe savedRecipe = recipeService.addRecipeForUser(recipe, userId);
        return ResponseEntity.ok(savedRecipe);
    }
}
