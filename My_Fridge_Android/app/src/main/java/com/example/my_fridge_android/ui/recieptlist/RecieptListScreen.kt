package com.example.my_fridge_android.ui.recieptlist

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.tooling.preview.PreviewParameter
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.my_fridge_android.data.source.remote.FridgeItem
import com.example.my_fridge_android.ui.components.EmptyScreen
import com.example.my_fridge_android.ui.components.LoadingBar
import com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiAction
import com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiEffect
import com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiState
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emptyFlow

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RecieptListScreen(
    uiState: UiState,
    uiEffect: Flow<UiEffect>,
    onAction: (UiAction) -> Unit,
) {
    var isVisible by remember { mutableStateOf(false) }
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(Unit) {
        delay(150)
        isVisible = true
    }

    // Handle UI effects
    LaunchedEffect(uiEffect) {
        uiEffect.collect { effect ->
            when (effect) {
                is UiEffect.NavigateToHome -> {
                    // Navigation handled by parent
                }
                is UiEffect.OpenCamera -> {
                    // Camera opening handled by parent
                }
                is UiEffect.OpenGallery -> {
                    // Gallery opening handled by parent
                }
                is UiEffect.ShowError -> {
                    snackbarHostState.showSnackbar(
                        message = effect.message,
                        duration = SnackbarDuration.Long
                    )
                }
                is UiEffect.ShowMessage -> {
                    snackbarHostState.showSnackbar(
                        message = effect.message,
                        duration = SnackbarDuration.Short
                    )
                }
            }
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.2f),
                        MaterialTheme.colorScheme.surface
                    )
                )
            )
    ) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            // Top bar with back button
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(
                    onClick = { onAction(UiAction.NavigateToHome) },
                    modifier = Modifier
                        .clip(CircleShape)
                        .background(MaterialTheme.colorScheme.primaryContainer)
                ) {
                    Icon(
                        imageVector = Icons.Default.ArrowBack,
                        contentDescription = "Back to Home",
                        tint = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                }
                
                Spacer(modifier = Modifier.width(16.dp))
                
                Text(
                    text = "My Fridge",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    style = MaterialTheme.typography.headlineMedium
                )
            }

            // Content
            when {
                uiState.isLoading -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            LoadingBar()
                            if (uiState.uploadStatus != null) {
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    text = uiState.uploadStatus,
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                            }
                        }
                    }
                }

                uiState.fridgeItems.isEmpty() -> {
                    Column {
                        // Show upload status if available
                        if (uiState.uploadStatus != null) {
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                colors = CardDefaults.cardColors(
                                    containerColor = if (uiState.uploadStatus.startsWith("✅"))
                                        MaterialTheme.colorScheme.primaryContainer
                                    else if (uiState.uploadStatus.startsWith("❌"))
                                        MaterialTheme.colorScheme.errorContainer
                                    else MaterialTheme.colorScheme.surfaceVariant
                                )
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(16.dp),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = uiState.uploadStatus,
                                        style = MaterialTheme.typography.bodyMedium,
                                        modifier = Modifier.weight(1f)
                                    )
                                    TextButton(
                                        onClick = { onAction(UiAction.ClearUploadStatus) }
                                    ) {
                                        Text("✕")
                                    }
                                }
                            }
                        }

                        AnimatedVisibility(
                            visible = isVisible,
                            enter = fadeIn(animationSpec = tween(600)) + slideInVertically(
                                animationSpec = spring(
                                    dampingRatio = Spring.DampingRatioMediumBouncy,
                                    stiffness = Spring.StiffnessMedium
                                ),
                                initialOffsetY = { it / 3 }
                            ),
                            exit = fadeOut(animationSpec = tween(300)) + slideOutVertically(),
                            modifier = Modifier.weight(1f)
                        ) {
                            EmptyFridgeContent()
                        }
                    }
                }
                else -> {
                    Column {
                        // Show upload status if available
                        if (uiState.uploadStatus != null) {
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                colors = CardDefaults.cardColors(
                                    containerColor = if (uiState.uploadStatus.startsWith("✅"))
                                        MaterialTheme.colorScheme.primaryContainer
                                    else if (uiState.uploadStatus.startsWith("❌"))
                                        MaterialTheme.colorScheme.errorContainer
                                    else MaterialTheme.colorScheme.surfaceVariant
                                )
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(16.dp),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = uiState.uploadStatus,
                                        style = MaterialTheme.typography.bodyMedium,
                                        modifier = Modifier.weight(1f)
                                    )
                                    TextButton(
                                        onClick = { onAction(UiAction.ClearUploadStatus) }
                                    ) {
                                        Text("✕")
                                    }
                                }
                            }
                        }

                        AnimatedVisibility(
                            visible = isVisible,
                            enter = fadeIn(animationSpec = tween(700)) + slideInVertically(
                                animationSpec = spring(
                                    dampingRatio = Spring.DampingRatioMediumBouncy,
                                    stiffness = Spring.StiffnessMedium
                                ),
                                initialOffsetY = { it / 3 }
                            ),
                            exit = fadeOut(animationSpec = tween(300)) + slideOutVertically(),
                            modifier = Modifier.weight(1f)
                        ) {
                            FridgeItemsList(
                                fridgeItems = uiState.fridgeItems,
                                onDeleteItem = { itemId ->
                                    onAction(UiAction.DeleteReceipt(itemId))
                                },
                                onIncreaseItem = { itemId ->
                                    onAction(UiAction.IncreaseItemAmount(itemId))
                                },
                                onDecreaseItem = { itemId ->
                                    onAction(UiAction.DecreaseItemAmount(itemId))
                                },
                                isDeleting = uiState.isDeleting,
                                deletingItemId = uiState.deletingItemId
                            )
                        }
                    }
                }
            }
        }

        // Floating Action Buttons
        Row(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            FloatingActionButton(
                onClick = { onAction(UiAction.ShowImagePicker) },
                containerColor = MaterialTheme.colorScheme.primary,
                contentColor = MaterialTheme.colorScheme.onPrimary
            ) {
                Icon(
                    imageVector = Icons.Default.Add,
                    contentDescription = "Add Receipt"
                )
            }
            
            ExtendedFloatingActionButton(
                onClick = { onAction(UiAction.ShowAddIngredients) },
                containerColor = MaterialTheme.colorScheme.secondary,
                contentColor = MaterialTheme.colorScheme.onSecondary
            ) {
                Text("Add Ingredients")
            }
        }

        // Snackbar host
        SnackbarHost(
            hostState = snackbarHostState,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 100.dp) // Add padding to avoid FABs
        )

        // Image Picker Bottom Sheet
        if (uiState.showImagePickerSheet) {
            ImagePickerBottomSheet(
                uiState = uiState,
                onDismiss = { onAction(UiAction.HideImagePicker) },
                onTakePhoto = { onAction(UiAction.TakePhoto) },
                onSelectFromGallery = { onAction(UiAction.SelectFromGallery) }
            )
        }
        
        // Add Ingredients Bottom Sheet
        if (uiState.showAddIngredientsSheet) {
            AddIngredientsBottomSheet(
                onDismiss = { onAction(UiAction.HideAddIngredients) },
                onAddIngredient = { name, amount, amountType ->
                    onAction(UiAction.AddIngredient(name, amount, amountType))
                }
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ImagePickerBottomSheet(
    uiState: UiState,
    onDismiss: () -> Unit,
    onTakePhoto: () -> Unit,
    onSelectFromGallery: () -> Unit
) {
    val bottomSheetState = rememberModalBottomSheetState(
        skipPartiallyExpanded = true
    )

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = bottomSheetState,
        dragHandle = { BottomSheetDefaults.DragHandle() },
        containerColor = MaterialTheme.colorScheme.surface,
        contentColor = MaterialTheme.colorScheme.onSurface
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
                .padding(bottom = 32.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Title
            Text(
                text = "Add Receipt",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.padding(bottom = 8.dp)
            )
            
            Text(
                text = "Choose how you'd like to add your receipt",
                fontSize = 16.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                modifier = Modifier.padding(bottom = 16.dp)
            )

            // Take Photo Option
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(enabled = !uiState.isCameraLaunching) {
                        onTakePhoto()
                        onDismiss()
                    },
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = if (uiState.isCameraLaunching) 
                        MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.6f)
                    else 
                        MaterialTheme.colorScheme.primaryContainer
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(20.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    if (uiState.isCameraLaunching) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(32.dp),
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    } else {
                        Text(
                            text = "📷",
                            fontSize = 32.sp
                        )
                    }
                    
                    Column(
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(
                            text = if (uiState.isCameraLaunching) "Opening Camera..." else "Take Photo",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                        Text(
                            text = if (uiState.isCameraLaunching) 
                                "Please wait..."
                            else 
                                "Use camera to capture receipt",
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                        )
                    }
                }
            }

            // Select from Gallery Option
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        onSelectFromGallery()
                        onDismiss()
                    },
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(20.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text(
                        text = "🖼️",
                        fontSize = 32.sp
                    )
                    
                    Column(
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(
                            text = "Choose from Gallery",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        Text(
                            text = "Select existing photo from gallery",
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AddIngredientsBottomSheet(
    onDismiss: () -> Unit,
    onAddIngredient: (String, String, String) -> Unit
) {
    val bottomSheetState = rememberModalBottomSheetState(
        skipPartiallyExpanded = true
    )
    
    var ingredientName by remember { mutableStateOf("") }
    var amount by remember { mutableStateOf("") }
    var selectedAmountType by remember { mutableStateOf("G") }
    var showDropdown by remember { mutableStateOf(false) }
    
    val amountTypes = listOf("L", "ml", "KG", "G", "amount")

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = bottomSheetState,
        dragHandle = { BottomSheetDefaults.DragHandle() },
        containerColor = MaterialTheme.colorScheme.surface,
        contentColor = MaterialTheme.colorScheme.onSurface
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
                .padding(bottom = 32.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(
                text = "Add Ingredient",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.padding(bottom = 8.dp)
            )
            
            OutlinedTextField(
                value = ingredientName,
                onValueChange = { ingredientName = it },
                label = { Text("Ingredient Name") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )
            
            OutlinedTextField(
                value = amount,
                onValueChange = { amount = it },
                label = { Text("Amount") },
                modifier = Modifier.fillMaxWidth(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                singleLine = true
            )
            
            ExposedDropdownMenuBox(
                expanded = showDropdown,
                onExpandedChange = { showDropdown = !showDropdown },
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedTextField(
                    value = selectedAmountType,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Amount Type") },
                    trailingIcon = {
                        Icon(
                            imageVector = Icons.Default.KeyboardArrowDown,
                            contentDescription = "Dropdown"
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                
                ExposedDropdownMenu(
                    expanded = showDropdown,
                    onDismissRequest = { showDropdown = false }
                ) {
                    amountTypes.forEach { type ->
                        DropdownMenuItem(
                            text = { Text(type) },
                            onClick = {
                                selectedAmountType = type
                                showDropdown = false
                            }
                        )
                    }
                }
            }
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                OutlinedButton(
                    onClick = onDismiss,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Cancel")
                }
                
                Button(
                    onClick = {
                        if (ingredientName.isNotBlank() && amount.isNotBlank()) {
                            onAddIngredient(ingredientName, amount, selectedAmountType)
                            onDismiss()
                        }
                    },
                    modifier = Modifier.weight(1f),
                    enabled = ingredientName.isNotBlank() && amount.isNotBlank()
                ) {
                    Text("Add")
                }
            }
        }
    }
}

@Composable
fun EmptyFridgeContent() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "🧊",
            fontSize = 120.sp
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Text(
            text = "Fridge is empty",
            fontSize = 24.sp,
            fontWeight = FontWeight.Medium,
            color = MaterialTheme.colorScheme.onSurface
        )
        
        Spacer(modifier = Modifier.height(8.dp))
        
        Text(
            text = "Tap the + button to add items to your fridge",
            fontSize = 16.sp,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
        )
    }
}

@Composable
fun FridgeItemsList(
    fridgeItems: List<FridgeItem>,
    onDeleteItem: (String) -> Unit,
    onIncreaseItem: (Int) -> Unit,
    onDecreaseItem: (Int) -> Unit,
    isDeleting: Boolean,
    deletingItemId: String?
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(fridgeItems) { item ->
            FridgeItemCard(
                item = item,
                onDelete = {
                    println("DEBUG: FridgeItemCard onDelete called for item: ${item.getSafeProduct().getSafeName()} with id: ${item.getSafeId()}")
                    onDeleteItem(item.getSafeId().toString())
                },
                onIncrease = { onIncreaseItem(item.getSafeId()) },
                onDecrease = { onDecreaseItem(item.getSafeId()) },
                isDeleting = isDeleting && item.getSafeId().toString() == deletingItemId
            )
        }

        item {
            Spacer(modifier = Modifier.height(120.dp)) // Space for FABs
        }
    }
}

@Composable
fun FridgeItemCard(
    item: FridgeItem,
    onDelete: () -> Unit,
    onIncrease: () -> Unit,
    onDecrease: () -> Unit,
    isDeleting: Boolean = false
) {
    // Calculate smart increment amount based on unit type
    val smartIncrement = when (item.getSafeProduct().getSafeCountType().lowercase()) {
        "g", "ml" -> 100
        "l", "amount", "kg" -> 1
        else -> 1
    }

    val canDecrease = item.getSafeAmount() > smartIncrement

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { /* Handle item click if needed */ },
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = item.getSafeProduct().getSafeName(),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSurface
                )

                Text(
                    text = "Amount: ${item.getSafeAmount()} ${item.getSafeProduct().getSafeCountType()}",
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )

                Text(
                    text = "Expires: ${item.getSafeSkt()}",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            // Amount control buttons
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                IconButton(
                    onClick = onDecrease,
                    enabled = canDecrease && !isDeleting
                ) {
                    if (isDeleting) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            color = MaterialTheme.colorScheme.primary
                        )
                    } else {
                        Text(
                            text = "-${smartIncrement}",
                            fontSize = 18.sp,
                            color = if (canDecrease) 
                                MaterialTheme.colorScheme.primary 
                            else 
                                MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
                        )
                    }
                }

                Text(
                    text = "${item.getSafeAmount()}",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.padding(horizontal = 8.dp)
                )

                IconButton(
                    onClick = onIncrease,
                    enabled = !isDeleting
                ) {
                    if (isDeleting) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            color = MaterialTheme.colorScheme.primary
                        )
                    } else {
                        Text(
                            text = "+${smartIncrement}",
                            fontSize = 18.sp,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }

                IconButton(
                    onClick = {
                        println("DEBUG: Delete button clicked for item: ${item.getSafeProduct().getSafeName()} with id: ${item.getSafeId()}")
                        onDelete()
                    },
                    enabled = !isDeleting
                ) {
                    if (isDeleting) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(24.dp),
                            color = MaterialTheme.colorScheme.error
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Default.Delete,
                            contentDescription = "Delete Item",
                            tint = MaterialTheme.colorScheme.error
                        )
                    }
                }
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun RecieptListScreenPreview(
    @PreviewParameter(RecieptListScreenPreviewProvider::class) uiState: UiState,
) {
    MaterialTheme {
        RecieptListScreen(
            uiState = uiState,
            uiEffect = emptyFlow(),
            onAction = {},
        )
    }
}
