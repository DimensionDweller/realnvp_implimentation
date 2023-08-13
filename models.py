import torch.nn.init as init

class Affine_Coupling(nn.Module):
    """
    Implements the affine coupling layer used in RealNVP.
    
    Args:
        mask (Tensor): Binary tensor defining the partitioning of input dimensions.
        hidden_dim (int): Dimension of hidden layers in scale and translation networks.
    """
    def __init__(self, mask, hidden_dim):
        super(Affine_Coupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim
        self.mask = nn.Parameter(mask, requires_grad=False)

        # Layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.scale)

        # Layers used to compute translation in affine transformation
        self.translation_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim)

    def _compute_scale(self, x):
        """
        Computes the scaling factor for the affine transformation.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Scaling factor.
        """
        s = torch.relu(self.scale_fc1(x * self.mask))
        s = torch.relu(self.scale_fc2(s))
        s = torch.relu(self.scale_fc3(s)) * self.scale
        return s

    def _compute_translation(self, x):
        """
        Computes the translation for the affine transformation.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Translation.
        """
        t = torch.relu(self.translation_fc1(x * self.mask))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)
        return t

    def forward(self, x):
        """
        Forward transformation from latent space to observed space.
        
        Args:
            x (Tensor): Latent space tensor.
            
        Returns:
            Tuple: Transformed tensor and log determinant of transformation.
        """
        s = self._compute_scale(x)
        t = self._compute_translation(x)
        y = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = torch.sum((1 - self.mask) * s, -1)
        return y, logdet

    def inverse(self, y):
        """
        Inverse transformation from observed space to latent space.
        
        Args:
            y (Tensor): Observed space tensor.
            
        Returns:
            Tuple: Transformed tensor and log determinant of inverse transformation.
        """
        s = self._compute_scale(y)
        t = self._compute_translation(y)
        x = self.mask * y + (1 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = torch.sum((1 - self.mask) * (-s), -1)
        return x, logdet


class RealNVP_2D(nn.Module):
    """
    A vanilla RealNVP class for modeling 2-dimensional distributions.

    Args:
        masks (list of lists): List of binary masks defining the partitioning of input dimensions for each affine coupling layer.
        hidden_dim (int): Dimension of hidden layers in the affine coupling layers.
    """

    def __init__(self, masks, hidden_dim):
        """
        Initializes the RealNVP_2D model with the given masks and hidden dimension.

        Args:
            masks (list of lists): List of binary masks.
            hidden_dim (int): Hidden dimension size.
        """
        super(RealNVP_2D, self).__init__()
        self.hidden_dim = hidden_dim
        self.masks = nn.ParameterList([nn.Parameter(torch.Tensor(m), requires_grad=False) for m in masks])
        self.affine_couplings = nn.ModuleList([Affine_Coupling(self.masks[i], self.hidden_dim) for i in range(len(self.masks))])

    def forward(self, x):
        """
        Forward transformation converting latent space variables into observed variables.

        Args:
            x (Tensor): Input tensor in latent space.

        Returns:
            Tuple: Transformed tensor in observed space and total log determinant of transformation.
        """
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot += logdet

        # Normalization layer to ensure observed variables are within the range of [-4, 4]
        logdet = torch.sum(torch.log(torch.abs(4 * (1 - (torch.tanh(y))**2))), -1)
        y = 4 * torch.tanh(y)
        logdet_tot += logdet
        
        return y, logdet_tot

    def inverse(self, y):
        """
        Inverse transformation converting observed variables into latent space variables.

        Args:
            y (Tensor): Input tensor in observed space.

        Returns:
            Tuple: Transformed tensor in latent space and total log determinant of inverse transformation.
        """
        x = y
        logdet_tot = 0

        # Inverse of the normalization layer
        logdet = torch.sum(torch.log(torch.abs(1.0/4.0 * 1/(1 - (x/4)**2))), -1)
        x = 0.5 * torch.log((1 + x/4) / (1 - x/4))
        logdet_tot += logdet

        # Inverse affine coupling layers
        for i in range(len(self.affine_couplings) - 1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot += logdet
            
        return x, logdet_tot
