package com.example.my_fridge_android.domain.repository

import android.net.Uri
import com.example.my_fridge_android.data.source.remote.FridgeItem
import com.example.my_fridge_android.data.source.remote.LoginRequest
import com.example.my_fridge_android.data.source.remote.RegisterRequest

interface MainRepository {
    suspend fun pingServer(): Result<String>
    suspend fun uploadReceiptImage(imageUri: Uri): Result<String>
    suspend fun registerUser(registerRequest: RegisterRequest): Result<String>
    suspend fun loginUser(loginRequest: LoginRequest): Result<String>
    suspend fun getFridgeItems(userId: Int): Result<List<FridgeItem>>
    suspend fun sendItemsToRecipeAssistant(items: List<FridgeItem>): Result<String>
    suspend fun sendRecipeSelection(originalResponse: String, userSelection: String): Result<String>
    suspend fun deleteProduct(userId: Int, productName: String): Result<String>
    suspend fun reduceAmount(
        userId: Int,
        productName: String,
        amount: Int,
        amountType: String
    ): Result<String>
    suspend fun addAmount(
        userId: Int,
        productName: String,
        categoryName: String,
        countType: String,
        price: Double,
        amount: Int,
        skt: String
    ): Result<String>

    suspend fun addManualIngredient(
        userId: Int,
        productName: String,
        amount: String,
        countType: String
    ): Result<String>
}
