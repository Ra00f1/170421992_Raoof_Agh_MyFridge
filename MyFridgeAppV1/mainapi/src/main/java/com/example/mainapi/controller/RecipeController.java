package com.example.mainapi.controller;

import com.example.mainapi.model.Recipe;
import com.example.mainapi.service.RecipeService;
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

    @PostMapping("/receive/{userId}")
    public ResponseEntity<?> receiveFromLlm(@PathVariable Long userId, @RequestBody Recipe recipe) 
    {
        try 
        {
            System.out.println("Tarif alındı:");
            System.out.println(recipe);
            recipeService.sendToDbApi(recipe, userId);
            return ResponseEntity.ok("Tarif DBAPI'ye yönlendirildi ve eksiltme işlemi başlatıldı.");
        } 
        catch (Exception e) 
        {
            System.out.println("HATA OLUŞTU: " + e.getMessage());
            return ResponseEntity.status(500).body("Hata oluştu: " + e.getMessage());
        }
    }
}
