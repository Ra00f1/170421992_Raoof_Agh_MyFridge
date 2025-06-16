package com.example.llmapi.controller;

    import com.example.llmapi.model.GeneratedRecipe;
import com.example.llmapi.service.RecipeService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/llm")
public class RecipeController 
{

    private final RecipeService service;

    public RecipeController(RecipeService service) 
    {
        this.service = service;
    }

    @GetMapping("/fridge/{userId}")
    public String getFridgeContent(@PathVariable int userId) 
    {
        return service.getUserFridge(userId);
    }

    @PostMapping("/saveAndSend/{userId}")
    public String saveAndSendLlmRecipe(@PathVariable int userId, @RequestBody GeneratedRecipe recipe) 
    {
        service.saveLlmRecipeLocally(userId, recipe);
        service.sendRecipeToMainApi(recipe, userId);
        return "Tarif kaydedildi ve MainAPI'ye gönderildi.";
    }

}

